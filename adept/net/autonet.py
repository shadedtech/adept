from __future__ import annotations

from collections import defaultdict
from copy import copy
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple

import networkx as nx
import torch
from torch import nn, Tensor

from adept.alias import Observation, Spec, Shape, HiddenStates
from adept.config import configurable
from adept.module import NetMod
from adept.util import import_util, torch_util, spec
from adept.util.space import Space, Discrete, Box
from adept.net.net1d import OutputLayer1D
from adept.net.net2d import OutputLayer2D
from adept.net.net3d import OutputLayer3D
from adept.net.net4d import OutputLayer4D

import logging

from adept.util.spec import to_dict

logger = logging.getLogger(__name__)

# TODO Handle hidden states

NodeID = str
ModuleID = str


DEFAULT_AUTO_SPEC = {
  "source1d": "adept.net.net1d.LinearNet",
  "source2d": "adept.net.net2d.SeqConvNet",
  "source3d": "adept.net.net3d.ImageConvNet",
  "source4d": "adept.net.net4d.Identity4D",
  "body": "adept.net.net1d.LSTM",
  "head1d": "adept.net.net1d.Identity1D",
  "head2d": "adept.net.net2d.Identity2D",
  "head3d": "adept.net.net3d.Identity3D",
  "head4d": "adept.net.net4d.Identity4D"
}


class AutoNetwork(nn.Module):
    @configurable
    def __init__(
        self,
        observation_spec: Spec,
        output_spec: Spec,
        auto_spec: dict[NodeID, ModuleID] = None,
        nodes: dict[NodeID, ModuleID] = None,
        edges: dict[NodeID, List[NodeID]] = None,
    ):
        super().__init__()
        if auto_spec is None:
            auto_spec = DEFAULT_AUTO_SPEC
        self._observation_spec = spec.to_dict(observation_spec, "obs")
        self._output_spec = spec.to_dict(output_spec, "action")
        self._auto_spec = auto_spec
        self._nodes = nodes
        self._edges = edges
        if not self._nodes or not self._edges:
            self._nodes, self._edges = self._from_auto(self._auto_spec)
        graph = nx.DiGraph(self._edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise Exception("Network graph must be a DAG")
        self._node_order = list(nx.topological_sort(graph))
        self._predecessors = {
            node_name: list(graph.predecessors(node_name))
            for node_name in self._node_order
        }
        self._successors = {
            node_name: list(graph.successors(node_name))
            for node_name in self._node_order
        }
        self._successor_counts = {
            node_name: len(successors)
            for node_name, successors in self._successors.items()
        }
        for node_name, netmod in self._get_netmods().items():
            self.add_module(node_name, netmod)

        for node_name, layer in self._get_output_layers().items():
            self.add_module(node_name, layer)
        logger.info(
            "Network initialized with %.2fM parameters" % (torch_util.get_num_params(self) / 1e6)
        )

    def forward(
        self, obs: Observation, hidden_states: HiddenStates, **kwargs
    ) -> Tuple[dict[NodeID, Tensor], HiddenStates]:
        obs = spec.to_dict(obs, name="obs")
        cache = {}
        counts = copy(self._successor_counts)
        nxt_hid = {}
        for cur in self._node_order:
            if cur in self._nodes or cur in self._output_spec:
                inputs = []
                marked_for_delete = []
                for p in self._predecessors[cur]:
                    if p in self._observation_spec:
                        inputs.append(obs[p])
                    elif p in cache:
                        shape = self._modules[p].output_shape(self._modules[cur].dim())
                        inputs.append(cache[p].view(-1, *shape))
                        counts[p] -= 1
                        if counts[p] == 0:
                            marked_for_delete.append(p)
                    else:
                        raise Exception("Unreachable")
                x = torch.cat(inputs, dim=1)
                hstate = None if cur not in hidden_states else hidden_states[cur]
                cache[cur], nxt_hid[cur] = self._modules[cur].forward(
                    x, hstate
                )
                for p in marked_for_delete:
                    del cache[p]
        nxt_hid = {k: v for k, v in nxt_hid.items() if v is not None}
        return cache, nxt_hid

    def new_hidden_states(
        self, device: torch.device, batch_sz: int = 1
    ) -> HiddenStates:
        out = {}
        for node_name, mod in self._modules.items():
            if isinstance(mod, NetMod):
                hiddens = mod.new_hidden_states(device, batch_sz)
                if hiddens is not None:
                    out[node_name] = hiddens
        return out

    def netmod_shapes(self) -> Iterator[Tuple[str, Shape, Shape]]:
        for node_name, mod in self._modules.items():
            if isinstance(mod, NetMod):
                yield node_name, mod.input_shape, mod.output_shape()

    def _from_auto(
        self, auto_spec: dict[NodeID, ModuleID]
    ) -> Tuple[dict[NodeID, ModuleID], dict[NodeID, List[NodeID]]]:
        input_dims = [len(space._non_batch_shape) for space in self._observation_spec.values()]
        output_dims = [len(space._non_batch_shape) for space in self._output_spec.values()]
        if len(input_dims) != len(set(input_dims)):
            raise Exception("Not implemented")  # TODO
        nodes = {}  # node_name to module mapping
        edges = defaultdict(list)  # graph data structure
        for input_node, shape in self._observation_spec.items():
            dim = len(shape)
            source_node = f"source{dim}d"
            edges[input_node].append(source_node)
            nodes[source_node] = auto_spec[source_node]
            edges[source_node].append("body")
        nodes["body"] = auto_spec["body"]
        # TODO kill from here down
        for dim in output_dims:
            head_node = f"head{dim}d"
            nodes[head_node] = auto_spec[head_node]
            edges["body"].append(head_node)
        for output_node, shape in self._output_spec.items():
            dim = len(shape)
            head_node = f"head{dim}d"
            edges[head_node].append(output_node)
        return nodes, edges

    def _get_netmods(self) -> Dict[str, NetMod]:
        netmods = {}
        for node_name in self._node_order:
            if node_name in self._nodes:
                nm_cls = import_util.import_object(self._nodes[node_name])
                shapes = []
                for p in self._predecessors[node_name]:
                    if p in self._observation_spec:
                        # TODO method to cast dimension
                        shapes.append(self._observation_spec[p])
                    elif p in netmods:
                        shapes.append(netmods[p].logit_shape(nm_cls.dim()))
                    elif p in self._output_spec:
                        shapes.append(self._output_spec[p])
                # non-feature dims must match
                in_shape = _merge_shapes(shapes)
                if nm_cls.is_configurable:
                    netmods[node_name] = nm_cls(node_name, in_shape, tag=node_name)
                else:
                    netmods[node_name] = nm_cls(node_name, in_shape)
        return netmods

    def _get_output_layers(self) -> Dict[str, nn.Module]:
        layers = {}
        for node_name, out_shape in self._output_spec.items():
            for p in self._predecessors[node_name]:
                shapes = []
                if p in self._observation_spec:
                    # TODO method to cast dimension
                    shapes.append(self._observation_spec[p])
                elif p in self._modules:
                    shapes.append(self._modules[p].output_shape(len(out_shape)))
                elif p in self._output_spec:
                    # TODO method to cast dimension
                    shapes.append(self._output_spec[p])
                in_shape = _merge_shapes(shapes)
                layers[node_name] = _get_output_layer(node_name, in_shape, out_shape)
        return layers


def _merge_shapes(shapes: List[Shape]) -> Shape:
    first_shape = shapes[0]
    if len(shapes) == 1:
        return first_shape
    tail_shapes = shapes[1:]
    if not all([len(t_shape) == len(first_shape) for t_shape in tail_shapes]):
        raise Exception("All shapes must have same dim")
    feat_dim = first_shape[0]
    non_feat_dims = list(first_shape[1:])
    for cur_shp in tail_shapes:
        feat_dim += cur_shp[0]
        for i, cur_non_feat_dims in enumerate(cur_shp[1:]):
            assert (
                cur_non_feat_dims == 1
                or cur_non_feat_dims == non_feat_dims[i]
                or non_feat_dims[i] == 1
            )
            non_feat_dims[i] = max(non_feat_dims[i], cur_non_feat_dims)
    return (feat_dim,) + tuple(non_feat_dims)


def _get_output_layer(name: str, input_shape: Shape, output_shape: Shape) -> nn.Module:
    dim = len(output_shape)
    if dim == 1:
        layer = OutputLayer1D(name, input_shape, output_shape)
    elif dim == 2:
        layer = OutputLayer2D(name, input_shape, output_shape)
    elif dim == 3:
        layer = OutputLayer3D(name, input_shape, output_shape)
    elif dim == 4:
        layer = OutputLayer4D(name, input_shape, output_shape)
    else:
        raise Exception(f"Invalid dim: {dim}")
    return layer


# rules
# non feature dims must match (or be broadcastable)
# must connect observation space and output space
if __name__ == "__main__":
    from adept.util import log_util

    log_util.setup_logging(logger)

    observation_shapes = {
        "screen": (3, 84, 84),
    }
    output_shapes = {
        "action": (1,),
    }
    net = AutoNetwork(
        observation_shapes,
        output_shapes
    )
    print(net._nodes)
    print(net._edges)
    print(net._modules)
    for info in net.netmod_shapes():
        print(info)

    obs_batch = {"screen": torch.zeros(4, 3, 84, 84)}
    hidden_states = defaultdict(lambda: torch.tensor([]))  # TODO should this be none?
    hidden_states["body"] = torch.zeros(4, 2, 512)
    stuff = net.forward(obs_batch, hidden_states)
    print(stuff)
