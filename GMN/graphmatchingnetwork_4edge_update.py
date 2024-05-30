from graphembeddingnetwork import GraphEmbeddingNet
from graphembeddingnetwork import GraphPropLayer
import torch
import torch.nn as nn


def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


def pairwise_dot_product_similarity(x, y):
    """Compute the dot product similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j.

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise dot product similarity.
    """
    return torch.mm(x, torch.transpose(y, 1, 0))


def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.

    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).

    Args:
      x: NxD float tensor.
      y: MxD float tensor.

    Returns:
      s: NxM float tensor, the pairwise cosine similarity.
    """
    x = torch.div(x, torch.sqrt(torch.max(torch.sum(x ** 2), 1e-12)))
    y = torch.div(y, torch.sqrt(torch.max(torch.sum(y ** 2), 1e-12)))
    return torch.mm(x, torch.transpose(y, 1, 0))


PAIRWISE_SIMILARITY_FUNCTION = {
    'euclidean': pairwise_euclidean_similarity,
    'dotproduct': pairwise_dot_product_similarity,
    'cosine': pairwise_cosine_similarity,
}


def get_pairwise_similarity(name):
    """Get pairwise similarity metric by name.

    Args:
      name: string, name of the similarity metric, one of {dot-product, cosine,
        euclidean}.

    Returns:
      similarity: a (x, y) -> sim function.

    Raises:
      ValueError: if name is not supported.
    """
    if name not in PAIRWISE_SIMILARITY_FUNCTION:
        raise ValueError('Similarity metric name "%s" not supported.' % name)
    else:
        return PAIRWISE_SIMILARITY_FUNCTION[name]


def compute_cross_attention(x, y, sim):
    """Compute cross attention.

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))
    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i

    Args:
      x: NxD float tensor.
      y: MxD float tensor.
      sim: a (x, y) -> similarity function.

    Returns:
      attention_x: NxD float tensor.
      attention_y: NxD float tensor.
    """
    a = sim(x, y)
    a_x = torch.softmax(a, dim=1)  # i->j
    a_y = torch.softmax(a, dim=0)  # j->i
    attention_x = torch.mm(a_x, y)
    attention_y = torch.mm(torch.transpose(a_y, 1, 0), x)
    return attention_x, attention_y


def batch_block_pair_attention(data,
                               block_idx,
                               n_blocks,
                               similarity='dotproduct'):
    """Compute batched attention between pairs of blocks.

    This function partitions the batch data into blocks according to block_idx.
    For each pair of blocks, x = data[block_idx == 2i], and
    y = data[block_idx == 2i+1], we compute

    x_i attend to y_j:
    a_{i->j} = exp(sim(x_i, y_j)) / sum_j exp(sim(x_i, y_j))
    y_j attend to x_i:
    a_{j->i} = exp(sim(x_i, y_j)) / sum_i exp(sim(x_i, y_j))

    and

    attention_x = sum_j a_{i->j} y_j
    attention_y = sum_i a_{j->i} x_i.

    Args:
      data: NxD float tensor.
      block_idx: N-dim int tensor.
      n_blocks: integer.
      similarity: a string, the similarity metric.

    Returns:
      attention_output: NxD float tensor, each x_i replaced by attention_x_i.

    Raises:
      ValueError: if n_blocks is not an integer or not a multiple of 2.
    """
    if not isinstance(n_blocks, int):
        raise ValueError('n_blocks (%s) has to be an integer.' % str(n_blocks))

    if n_blocks % 2 != 0:
        raise ValueError('n_blocks (%d) must be a multiple of 2.' % n_blocks)

    sim = get_pairwise_similarity(similarity)

    results = []

    # This is probably better than doing boolean_mask for each i
    partitions = []
    for i in range(n_blocks):
        partitions.append(data[block_idx == i, :])

    for i in range(0, n_blocks, 2):
        x = partitions[i]
        y = partitions[i + 1]
        attention_x, attention_y = compute_cross_attention(x, y, sim)
        results.append(attention_x)
        results.append(attention_y)
    results = torch.cat(results, dim=0)

    return results


def graph_prop_once_4edge(edge_states,
                    from_idx,
                    to_idx,
                    message_net,
                    aggregation_module=None,
                    node_features=None):
    """
    for edge
    """
    edge_inputs = []

    if node_features is not None:
        from_states = node_features[from_idx]
        to_states = node_features[to_idx]
        edge_inputs = [from_states, to_states]

    edge_inputs.append(edge_states)

    edge_inputs = torch.cat(edge_inputs, dim=-1)
    messages = message_net(edge_inputs)

    return messages

class GraphPropMatchingLayer(GraphPropLayer):
    """A graph propagation layer that also does cross graph matching.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """
    # oyxy加
    def __init__(self,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 edge_net_init_scale=0.1,
                 edge_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 prop_type='embedding',
                 name='graph-net'):

        print("init graph_matching_network_4edge_update")
        self._edge_update_type = edge_update_type

        super(GraphPropMatchingLayer, self).__init__(
            node_state_dim,
            edge_state_dim,
            edge_hidden_sizes,
            node_hidden_sizes,
            edge_net_init_scale=edge_net_init_scale,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            layer_norm=layer_norm,
            prop_type=prop_type,
            name=name
        )

    def build_model(self):
        layer = []
        layer.append(nn.Linear(self._node_state_dim*2 + self._edge_state_dim, self._edge_hidden_sizes[0]))
        for i in range(1, len(self._edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
        self._message_net = nn.Sequential(*layer)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                layer.append(nn.Linear(self._node_state_dim*2 + self._edge_state_dim, self._edge_hidden_sizes[0]))
                for i in range(1, len(self._edge_hidden_sizes)):
                    layer.append(nn.ReLU())
                    layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
                self._reverse_message_net = nn.Sequential(*layer)
            else:
                self._reverse_message_net = self._message_net

        if self._edge_update_type == 'gru':
            if self._prop_type == 'embedding':
                self.GRU = torch.nn.GRU(self._edge_state_dim * 2, self._edge_state_dim)
            elif self._prop_type == 'matching':
                self.GRU = torch.nn.GRU(self._edge_state_dim * 3, self._edge_state_dim)
        else:
            layer = []
            if self._prop_type == 'embedding':
                layer.append(nn.Linear(self._edge_state_dim * 3, self._edge_hidden_sizes[0]))
            elif self._prop_type == 'matching':
                layer.append(nn.Linear(self._edge_state_dim * 4, self._edge_hidden_sizes[0]))
            for i in range(1, len(self._edge_hidden_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._edge_hidden_sizes[i - 1], self._edge_hidden_sizes[i]))
            self.MLP = nn.Sequential(*layer)

    def _compute_aggregated_messages_4edge(
            self, edge_states, from_idx, to_idx, node_features=None):
        """Compute aggregated messages for each node.

        Args:
          edge_states: [n_edges, input_edge_state_dim] float tensor, edge states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          node_features: if not None, should be [n_nodes, node_embedding_dim]
            tensor, node features.

        Returns:
          aggregated_messages: [n_edges, aggregated_message_dim] float tensor, the
            aggregated messages for each edge.
        """

        aggregated_messages = graph_prop_once_4edge(
            edge_states,
            from_idx,
            to_idx,
            self._message_net,
            aggregation_module=None,
            node_features=node_features)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            reverse_aggregated_messages = graph_prop_once_4edge(
                edge_states,
                to_idx,
                from_idx,
                self._reverse_message_net,
                aggregation_module=None,
                node_features=node_features)

            aggregated_messages += reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)

        return aggregated_messages

    def _compute_edge_update(self,
                             edge_states,
                             edge_state_inputs,
                             edge_features=None):
        if self._edge_update_type in ('mlp', 'residual'):
            edge_state_inputs.append(edge_states)
        if edge_features is not None:
            edge_state_inputs.append(edge_features)

        if len(edge_state_inputs) == 1:
            edge_state_inputs = edge_state_inputs[0]
        else:
            edge_state_inputs = torch.cat(edge_state_inputs, dim=-1)

        if self._edge_update_type == 'gru':
            edge_state_inputs = torch.unsqueeze(edge_state_inputs, 0)
            edge_states = torch.unsqueeze(edge_states, 0)
            _, new_edge_states = self.GRU(edge_state_inputs, edge_states)
            new_edge_states = torch.squeeze(new_edge_states)
            return new_edge_states
        else:
            mlp_output = self.MLP(edge_state_inputs)
            if self._layer_norm:
                mlp_output = nn.self.layer_norm2(mlp_output)
            if self._edge_update_type == 'mlp':
                return mlp_output
            elif self._edge_update_type == 'residual':
                return edge_states + mlp_output
            else:
                raise ValueError('Unknown edge update type %s' % self._edge_update_type)

    def forward(self,
                edge_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                similarity='dotproduct',
                edge_features=None,
                node_features=None):
        """Run one propagation step with cross-graph matching.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.
          graph_idx: [n_onodes] int tensor, graph id for each node.
          n_graphs: integer, number of graphs in the batch.
          similarity: type of similarity to use for the cross graph attention.
          edge_features: if not None, should be [n_edges, edge_feat_dim] tensor,
            extra edge features.
          node_features: if not None, should be [n_nodes, node_feat_dim] tensor,
            extra node features.

        Returns:
          node_states: [n_nodes, node_state_dim] float tensor, new node states.

        Raises:
          ValueError: if some options are not provided correctly.
        """
        aggregated_messages = self._compute_aggregated_messages_4edge(
            edge_states, from_idx, to_idx, node_features=node_features)

        cross_graph_attention = batch_block_pair_attention(
            edge_states, self.graph_idx_4edge, n_graphs, similarity=similarity)
        attention_input = edge_states - cross_graph_attention

        return self._compute_edge_update(edge_states,
                                         [aggregated_messages, attention_input],
                                         edge_features=edge_features)


class GraphAggregator(nn.Module):
    """This module computes graph representations by aggregating from parts."""

    def __init__(self,
                 edge_hidden_sizes,
                 graph_transform_sizes=None,
                 input_size=None,
                 gated=True,
                 aggregation_type='sum',
                 name='graph-aggregator'):
        """Constructor.

        Args:
          edge_hidden_sizes: the hidden layer sizes of the edge transformation nets.
            The last element is the size of the aggregated graph representation.

          graph_transform_sizes: sizes of the transformation layers on top of the
            graph representations.  The last element of this list is the final
            dimensionality of the output graph representations.

          gated: set to True to do gated aggregation, False not to.

          aggregation_type: one of {sum, max, mean, sqrt_n}.
          name: name of this module.
        """
        super(GraphAggregator, self).__init__()

        self._edge_hidden_sizes = edge_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = edge_hidden_sizes[-1]
        self._input_size = input_size
        #  The last element is the size of the aggregated graph representation.
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = None
        self.MLP1, self.MLP2 = self.build_model()

    def build_model(self):
        edge_hidden_sizes = self._edge_hidden_sizes
        if self._gated:
            edge_hidden_sizes[-1] = self._graph_state_dim * 2

        layer = []
        layer.append(nn.Linear(self._input_size[0], edge_hidden_sizes[0]))
        for i in range(1, len(edge_hidden_sizes)):
            layer.append(nn.ReLU())
            layer.append(nn.Linear(edge_hidden_sizes[i - 1], edge_hidden_sizes[i]))
        MLP1 = nn.Sequential(*layer)

        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            layer = []
            layer.append(nn.Linear(self._graph_state_dim, self._graph_transform_sizes[0]))
            for i in range(1, len(self._graph_transform_sizes)):
                layer.append(nn.ReLU())
                layer.append(nn.Linear(self._graph_transform_sizes[i - 1], self._graph_transform_sizes[i]))
            MLP2 = nn.Sequential(*layer)

        return MLP1, MLP2

    def forward(self, edge_states, graph_idx_4edge, n_graphs):
        """Compute aggregated graph representations.

        Args:
          edge_states: [n_edges, edge_state_dim] float tensor, edge states of a
            batch of graphs concatenated together along the first dimension.
          graph_idx_4edge: [n_edges] int tensor, graph ID for each edge.
          n_graphs: integer, number of graphs in this batch.

        Returns:
          graph_states: [n_graphs, graph_state_dim] float tensor, graph
            representations, one row for each graph.
        """

        edge_states_g = self.MLP1(edge_states)

        if self._gated:
            gates = torch.sigmoid(edge_states_g[:, :self._graph_state_dim])
            edge_states_g = edge_states_g[:, self._graph_state_dim:] * gates

        from segment import unsorted_segment_sum
        graph_states = unsorted_segment_sum(edge_states_g, graph_idx_4edge, n_graphs)

        if self._aggregation_type == 'max':
            # reset everything that's smaller than -1e5 to 0.
            graph_states *= torch.FloatTensor(graph_states > -1e5)
        # transform the reduced graph states further


        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            graph_states = self.MLP2(graph_states)

        return graph_states


class GraphMatchingNet(GraphEmbeddingNet):
    """Graph matching net.

    This class uses graph matching layers instead of the simple graph prop layers.

    It assumes the incoming graph data is batched and paired, i.e. graph 0 and 1
    forms the first pair and graph 2 and 3 are the second pair etc., and computes
    cross-graph attention-based matching for each pair.
    """

    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 edge_state_dim,
                 edge_hidden_sizes,
                 node_hidden_sizes,
                 n_prop_layers,
                 share_prop_params=False,
                 edge_net_init_scale=0.1,
                 edge_update_type='residual',
                 use_reverse_direction=True,
                 reverse_dir_param_different=True,
                 layer_norm=False,
                 layer_class=GraphPropLayer,
                 similarity='dotproduct',
                 prop_type='embedding'):
        self._edge_update_type = edge_update_type
        super(GraphMatchingNet, self).__init__(
            encoder,
            aggregator,
            node_state_dim,
            edge_state_dim,
            edge_hidden_sizes,
            node_hidden_sizes,
            n_prop_layers,
            share_prop_params=share_prop_params,
            edge_net_init_scale=edge_net_init_scale,
            use_reverse_direction=use_reverse_direction,
            reverse_dir_param_different=reverse_dir_param_different,
            layer_norm=layer_norm,
            layer_class=GraphPropMatchingLayer,
            prop_type=prop_type,
        )
        self._similarity = similarity

    def _build_layer(self, layer_id):
        """Build one layer in the network."""
        return self._layer_class(
            self._node_state_dim,
            self._edge_state_dim,
            self._edge_hidden_sizes,
            self._node_hidden_sizes,
            edge_net_init_scale=self._edge_net_init_scale,
            edge_update_type=self._edge_update_type,
            use_reverse_direction=self._use_reverse_direction,
            reverse_dir_param_different=self._reverse_dir_param_different,
            layer_norm=self._layer_norm,
            prop_type=self._prop_type)

    def _apply_layer(self,
                     layer,
                     edge_states,
                     from_idx,
                     to_idx,
                     graph_idx,
                     n_graphs,
                     node_features):
        """Apply one layer on the given inputs."""
        return layer(edge_states, from_idx, to_idx, graph_idx, n_graphs,
                     similarity=self._similarity, node_features=node_features)

    # oyxy加
    def forward(self,
                node_features,
                edge_features,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs):
        """Compute graph representations.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: [n_edges, edge_feat_dim] float tensor.
          from_idx: [n_edges] int tensor, index of the from node for each edge.
          to_idx: [n_edges] int tensor, index of the to node for each edge.
          graph_idx: [n_nodes] int tensor, graph id for each node.
          n_graphs: int, number of graphs in the batch.

        Returns:
          graph_representations: [n_graphs, graph_representation_dim] float tensor,
            graph representations.
        """

        node_features, edge_features = self._encoder(node_features, edge_features)
        node_states = node_features
        edge_states = edge_features

        for layer in self._prop_layers:
            # node_features could be wired in here as well, leaving it out for now as
            # it is already in the inputs
            edge_states = self._apply_layer(
                layer,
                edge_states,
                from_idx,
                to_idx,
                graph_idx,
                n_graphs,
                node_features)

        return self._aggregator(edge_states, self.graph_idx_4edge, n_graphs)

    # oyxy加
    def add_edge_attributes(self, graph_idx_4edge):
        self.graph_idx_4edge = graph_idx_4edge
        for layer in self._prop_layers:
            layer.graph_idx_4edge = graph_idx_4edge
