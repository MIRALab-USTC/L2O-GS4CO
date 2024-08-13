"""
File adapted from https://github.com/dandip/DSRPytorch
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

from settings.consts import SAFE_EPSILON
from dso_utils_graph.transformer_utils import TransformerDSOEncoder
                


class DSOAgent(nn.Module):
    def __init__(self, operators, min_length=4, max_length=128, hidden_size=128, num_layers=2, soft_length=64, two_sigma_square=16):
        super().__init__()

        self.input_size = 2 * (operators.operator_length+1) + operators.scatter_max_degree + 1 # One-hot encoded parent and sibling
        self.hidden_size = hidden_size
        self.output_size = operators.operator_length # Output is a softmax distribution over all operators
        self.num_layers = num_layers
        self.operators = operators

        # Initial cell optimization
        self.init_input = nn.Parameter(data=torch.rand(1, self.input_size), requires_grad=True)
        self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True)

        self.min_length = min_length
        self.max_length = max_length
        self.soft_length = soft_length
        self.two_sigma_square = two_sigma_square

        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            proj_size = self.output_size,
        )
        self.init_hidden_lstm = nn.Parameter(data=torch.rand(self.num_layers, 1, self.output_size), requires_grad=True)
        self.activation = nn.Softmax(dim=1)

    @torch.no_grad()
    def sample_sequence_eval(self, n):
        sequences = torch.zeros(n, 0, dtype=torch.long)

        input_tensor = self.init_input.expand(n, -1).contiguous()
        hidden_tensor = self.init_hidden.expand(-1, n, -1).contiguous()
        hidden_lstm = self.init_hidden_lstm.expand(-1, n, -1).contiguous()

        sequence_mask = torch.ones(n, dtype=torch.bool)
        counters = torch.ones(n, 1, dtype=torch.long) # Number of tokens that must be sampled to complete expression

        length = 0
        all_lengths = torch.zeros(n, dtype=torch.long)

        # While there are still tokens left for sequences in the batch
        all_log_prob_list, all_counters_list, all_inputs_list, all_scatter_degree_list, scatter_parent_where_seq = [], [], [], [torch.zeros((n,1), dtype=torch.long)], torch.full((n,1), fill_value=-1, dtype=torch.long)
        while(sequence_mask.any()):
            output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm, length)

            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, counters, length, sequences, scatter_degree=all_scatter_degree_list[-1], scatter_parent_where_seq=scatter_parent_where_seq)
            output = output / torch.sum(output, dim=1, keepdim=True)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            token = dist.sample()

            # Add sampled tokens to sequences
            sequences = torch.cat((sequences, token[:, None]), dim=1)
            length += 1
            all_lengths[sequence_mask] += 1

            # Add log probability of current token
            all_log_prob_list.append(dist.log_prob(token)[:, None])

            # Add entropy of current token
            all_counters_list.append(counters)
            all_inputs_list.append(input_tensor)

            # Update counter
            counters = counters + (torch.logical_and(self.operators.arity_one_begin<=token, token<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=token, token<self.operators.arity_two_end).long() * 2 - 1)[:, None]
            sequence_mask = torch.logical_and(sequence_mask, counters.squeeze(1) > 0)

            # Compute next parent and sibling; assemble next input tensor
            input_tensor, scatter_parent_where_seq = self.get_parent_sibling(n, sequences, length-1, sequence_mask, all_scatter_degree_list, scatter_parent_where_seq=scatter_parent_where_seq) # get input_tensor and update all_scatter_degree_list

        # Filter entropies log probabilities using the sequence_mask
        assert all_lengths.min() >= self.min_length and all_lengths.max() <= self.max_length+1 and all_lengths.max() == sequences.shape[1]
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return sequences, all_lengths, log_probs, (all_counters_list, all_inputs_list, all_scatter_degree_list, scatter_parent_where_seq)

    def sample_sequence_train(self, sequences, info_lists):
        all_counters_list, all_inputs_list, all_scatter_degree_list, scatter_parent_where_seq = info_lists
        assert sequences.shape[1] == len(all_counters_list) == len(all_inputs_list)
        n = len(sequences)

        all_inputs_list[0] = self.init_input.expand(n, -1).contiguous()
        hidden_tensor = self.init_hidden.expand(-1, n, -1).contiguous()
        hidden_lstm = self.init_hidden_lstm.expand(-1, n, -1).contiguous()

        all_log_prob_list, all_entropy_list = [], []
        for i, (token, counters, input_tensor, scatter_degree) in enumerate(zip(sequences.t(), all_counters_list, all_inputs_list, all_scatter_degree_list)):
            output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm, i)

            output = self.apply_constraints(output, counters, i, sequences[:,:i], scatter_degree=scatter_degree, scatter_parent_where_seq=scatter_parent_where_seq[:, :i+1])
            output = output / torch.sum(output, dim=1, keepdim=True)

            dist = torch.distributions.Categorical(output)
            all_log_prob_list.append(dist.log_prob(token)[:, None])
            all_entropy_list.append(dist.entropy()[:, None])

        entropies = torch.cat(all_entropy_list, dim=1)
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return entropies, log_probs


    def forward(self, input, hidden, hidden_lstm, cur_length):
        """Input should be [parent, sibling]
        """

        output, (hn, cn) = self.lstm(input.unsqueeze(0).float(), (hidden_lstm, hidden))
        output = output.squeeze(0)

        # ~ soft length constraint
        prior_vec = torch.zeros(self.output_size)
        if cur_length < self.soft_length:
            prior_vec[self.operators.arity_zero_begin:self.operators.arity_zero_end] = - (self.soft_length - cur_length) ** 2 / self.two_sigma_square
        elif cur_length > self.soft_length:
            prior_vec[self.operators.arity_two_begin:self.operators.arity_two_end] = - (cur_length - self.soft_length) ** 2 / self.two_sigma_square
        output = output + prior_vec[None, :]

        output = self.activation(output)
        return output, cn, hn

    def apply_constraints(self, output, counters, length, sequences, scatter_degree, scatter_parent_where_seq):
        """Applies in situ constraints to the distribution contained in output based on the current tokens
        """
        # Add small epsilon to output so that there is a probability of selecting
        # everything. Otherwise, constraints may make the only operators ones
        # that were initially set to zero, which will prevent us selecting
        # anything, resulting in an error being thrown
        
        output = output + SAFE_EPSILON

        # turn off column features if we set var_list = simple
        # if self.operators.column_var_mask is not None:
        #     output[:, self.operators.column_var_mask] = 0.

        # ~ Check that minimum length will be met ~
        # Explanation here
        min_boolean_mask = (counters + length) >= self.min_length
        min_length_mask = torch.logical_or(self.operators.nonzero_arity_mask, min_boolean_mask)
        output[~min_length_mask] = 0.

        # ~ Check that maximum length won't be exceed ~
        max_boolean_mask = (counters + length) < self.max_length
        max_length_mask = torch.logical_or(self.operators.zero_arity_mask, max_boolean_mask)
        output[~max_length_mask] = 0

        # forbid direct inverse function
        # last_token = sequences[:, -1]
        # output[xxx, ]


        # ~ Ensure that all expressions have a variable ~
        # nonvar_zeroarity_mask = ~torch.logical_and(self.operators.zero_arity_mask, self.operators.nonvariable_mask)
        if (length == 0): # First thing we sample can't be
            # output = torch.minimum(output, nonvar_zeroarity_mask)
            output[:, self.operators.const_mask.squeeze(0)] = 0
        else:
            last_counter_mask = (counters == 1)
            non_var_now_mask = torch.logical_not( (sequences < self.operators.variable_end).any(dim=1, keepdim=True) )
            last_token_and_no_var_mask = torch.logical_and(last_counter_mask, non_var_now_mask)
            const_and_last_token_and_no_var_mask = torch.logical_and(last_token_and_no_var_mask, self.operators.const_mask)
            output[const_and_last_token_and_no_var_mask] = 0
            # nonvar_zeroarity_mask = nonvar_zeroarity_mask.repeat(counters.shape[0], 1)
            # # Don't sample a nonvar zeroarity token if the counter is at 1 and
            # # we haven't sampled a variable yet
            # counter_mask = (counters == 1)
            # contains_novar_mask = ~(torch.isin(sequences, self.operators.variable_tensor).any(axis=1))
            # last_token_and_no_var_mask = (~torch.logical_and(counter_mask, contains_novar_mask)[:, None]).long()
            # nonvar_zeroarity_mask = torch.max(nonvar_zeroarity_mask, last_token_and_no_var_mask * torch.ones(nonvar_zeroarity_mask.shape)).long()
            # output = torch.minimum(output, nonvar_zeroarity_mask)

            # ~ forbid inverse unary
            last_token = sequences[:, -1]
            last_token_has_inverse = torch.where(self.operators.have_inverse[last_token])[0]
            last_token_inverse = self.operators.where_inverse[last_token[last_token_has_inverse]]
            output[last_token_has_inverse, last_token_inverse] = 0

            # degree 1,3,5,...
            scatter_mod = scatter_degree % 2
            where_same_sub_tree = scatter_parent_where_seq[:, :-1] == scatter_parent_where_seq[:,-1:]
            sub_tree_counter_is_degree_2 = torch.logical_and(torch.logical_and(sequences>=self.operators.arity_two_begin, sequences < self.operators.arity_two_end), where_same_sub_tree)
            sub_tree_counter_is_degree_0 = torch.logical_and(sequences < self.operators.arity_zero_end, where_same_sub_tree)
            sub_tree_counter = 1 + sub_tree_counter_is_degree_2.sum(dim=1, keepdim=True) - sub_tree_counter_is_degree_0.sum(dim=1, keepdim=True)
            sub_tree_last_counter_mask = (sub_tree_counter == 1)

            where_scatter_degree_1 = (scatter_mod == 1)
            where_scatter_degree_1_and_last_counter = torch.logical_and(where_scatter_degree_1, sub_tree_last_counter_mask)

            non_message_passing_1 = torch.logical_not( torch.logical_and(sequences < self.operators.variable_constraint_end, where_same_sub_tree).any(dim=1, keepdim=True) )

            where_scatter_degree_1_and_last_counter_and_non_message_passing = torch.logical_and(where_scatter_degree_1_and_last_counter, non_message_passing_1)
            scatter_degree_1_mask = torch.logical_and(where_scatter_degree_1_and_last_counter_and_non_message_passing, self.operators.scatter_degree_1_mask)
            output[scatter_degree_1_mask] = 0.

            # degree 2
            where_scatter_degree_2 = torch.logical_and(scatter_degree > 0, scatter_mod == 0)
            where_scatter_degree_2_and_last_counter = torch.logical_and(where_scatter_degree_2, sub_tree_last_counter_mask)
            non_message_passing_2 = torch.logical_not(torch.logical_and(torch.logical_and(sequences < self.operators.variable_variable_end, sequences >= self.operators.variable_variable_begin), where_same_sub_tree).any(dim=1, keepdim=True))

            where_scatter_degree_2_and_last_counter_and_non_message_passing = torch.logical_and(where_scatter_degree_2_and_last_counter, non_message_passing_2)
            scatter_degree_2_mask = torch.logical_and(where_scatter_degree_2_and_last_counter_and_non_message_passing, self.operators.scatter_degree_2_mask)
            output[scatter_degree_2_mask] = 0.

        # mask to avoid too many message passing layers
        scatter_should_mask = (scatter_degree >= self.operators.scatter_max_degree)
        scatter_mask = torch.logical_and(scatter_should_mask, self.operators.scatter_mask)
        output[scatter_mask] = 0.

        # mask constraint features when scatter_degree == 0
        where_scatter_degree_0 = (scatter_degree == 0)
        scatter_degree_0_mask = torch.logical_and(where_scatter_degree_0, self.operators.scatter_degree_0_mask)
        output[scatter_degree_0_mask] = 0.

        return output

    def get_parent_sibling(self, batch_size, sequences, recent, sequence_mask, all_scatter_degree_list, scatter_parent_where_seq):
        parent_sibling, parent_sibling_where = self._get_parent_sibling(batch_size, sequences, recent, sequence_mask)
        parent_where = parent_sibling_where[:,0]

        current_scatter_degree = (torch.as_tensor([all_scatter_degree_list[x][i] for i,x in enumerate(parent_where)], dtype=torch.long) + \
                                 (sequences[torch.arange(batch_size), parent_where] >= self.operators.scatter_begin).long())
        all_scatter_degree_list.append(current_scatter_degree[:, None])

        scatter_parent_where_now = scatter_parent_where_seq[torch.arange(batch_size), parent_where]
        scatter_parent_where_now[sequences[:, recent]>=self.operators.scatter_begin] = recent
        scatter_parent_where_seq = torch.cat((scatter_parent_where_seq, scatter_parent_where_now[:, None]), dim=1)

        input_tensor1 = F.one_hot(parent_sibling.reshape(-1), num_classes=self.output_size+1)
        input_tensor1 = input_tensor1.reshape(batch_size, -1)
        input_tensor2 = F.one_hot(current_scatter_degree, num_classes=self.operators.scatter_max_degree + 1)

        input_tensor = torch.cat((input_tensor1, input_tensor2), dim=1)

        return input_tensor, scatter_parent_where_seq


    def _get_parent_sibling(self, batch_size, sequences, recent, sequence_mask):
        """Returns parent, sibling for the most recent token in token_list
        """

        parent_sibling = torch.full(size=(batch_size, 2), fill_value=-1, dtype=torch.long)
        parent_sibling[~sequence_mask] = 0

        parent_sibling_where = torch.full(size=(batch_size,2), fill_value=-1, dtype=torch.long)
        parent_sibling_where[~sequence_mask] = 0

        token_last = sequences[:, recent]
        token_last_is_parent = self.operators.arity_tensor[token_last] > 0
        parent_sibling[token_last_is_parent, 0] = token_last[token_last_is_parent]
        parent_sibling[token_last_is_parent, 1] = self.output_size # means empty token

        parent_sibling_where[token_last_is_parent,0] = recent

        c = torch.zeros(batch_size, dtype=torch.long)
        for i in range(recent, -1, -1):
            unfinished_bool_index = (parent_sibling < 0).any(dim=1)
            if not unfinished_bool_index.any():
                break

            unfinished_token_i = sequences[:, i][unfinished_bool_index]
            c[unfinished_bool_index] += (torch.logical_and(self.operators.arity_one_begin<=unfinished_token_i, unfinished_token_i<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=unfinished_token_i, unfinished_token_i<self.operators.arity_two_end).long() * 2 - 1)
            found_now = torch.logical_and(unfinished_bool_index, c==0)

            parent_sibling[found_now] = sequences[found_now, i:(i+2)]
            parent_sibling_where[found_now,0] = i
            parent_sibling_where[found_now,1] = i+1

        assert (torch.logical_and(parent_sibling >= 0, parent_sibling <= self.output_size)).all()
        assert (parent_sibling_where[:,0] >= 0).all() # or (recent == 0)

        return parent_sibling, parent_sibling_where


class TransformerDSOAgent(DSOAgent):
    def __init__(self, operators, min_length=4, max_length=128, soft_length=64, two_sigma_square=16, d_model=64, num_heads=8, d_ff=256, num_layers=6, structural_encoding=True):
        super(DSOAgent, self).__init__()
        self.operators, self.min_length, self.max_length, self.soft_length, self.two_sigma_square = operators, min_length, max_length, soft_length, two_sigma_square
        self.output_size = operators.operator_length
        self.transformer = TransformerDSOEncoder(self.output_size, operators.scatter_max_degree, max_length=max_length, d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers, structural_encoding=structural_encoding)
        self.activation = nn.Softmax(dim=1)

    def forward(self, raw_x, scatter_degree, parentchild_indices, parent_child_now, silbing_indices, silbing_now):
        cur_length = raw_x.shape[1]-1
        output = self.transformer(raw_x, scatter_degree, parentchild_indices, parent_child_now, silbing_indices, silbing_now) # only return the last logits

        # ~ soft length constraint
        prior_vec = torch.zeros(self.output_size)

        if cur_length < self.soft_length:
            prior_vec[self.operators.arity_zero_begin:self.operators.arity_zero_end] = - (self.soft_length - cur_length) ** 2 / self.two_sigma_square
        elif cur_length > self.soft_length:
            prior_vec[self.operators.arity_two_begin:self.operators.arity_two_end] = - (cur_length - self.soft_length) ** 2 / self.two_sigma_square
        output = output + prior_vec[None, :]

        output = self.activation(output)
        return output

    @torch.no_grad()
    def sample_sequence_eval(self, n):
        max_len = self.max_length + 3

        sequences = torch.zeros((n,max_len), dtype=torch.long)
        sequences[:,0] = self.output_size
        scatter_degree = torch.zeros_like(sequences)
        parent_child_pairs = torch.zeros((n*(max_len), 3), dtype=torch.long)
        parent_child_length = torch.zeros(max_len, dtype=torch.long)
        silbing_pairs = torch.zeros((n*(max_len), 3), dtype=torch.long)
        silbing_length = torch.zeros(max_len, dtype=torch.long)
        scatter_parent_where_seq = torch.full_like(sequences, fill_value=-1)

        length = 0
        all_lengths = torch.zeros(n, dtype=torch.long)
        sequence_mask = torch.ones(n, dtype=torch.bool)
        counters = torch.ones(n, 1, dtype=torch.long) # Number of tokens that must be sampled to complete expression
        all_log_prob_list, all_counters_list = [], []

        while(sequence_mask.any()):

            output = self.forward(sequences[:,:(length+1)], scatter_degree[:,:(length+1)], parent_child_pairs[:parent_child_length[max(length-1,0)]], parent_child_pairs[parent_child_length[max(length-1,0)]:parent_child_length[length]], silbing_pairs[:silbing_length[max(length-1,0)]], silbing_pairs[silbing_length[max(length-1,0)]:silbing_length[length]])

            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, counters, length, sequences[:,1:(length+1)], scatter_degree=scatter_degree[:,length][:,None], scatter_parent_where_seq=scatter_parent_where_seq[:,:(length+1)])
            output = output / torch.sum(output, dim=1, keepdim=True)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            token = dist.sample()

            # Add sampled tokens to sequences
            sequences[:, length+1] = token
            length += 1
            all_lengths[sequence_mask] += 1

            # Add log probability of current token
            all_log_prob_list.append(dist.log_prob(token)[:, None])

            # Add entropy of current token
            all_counters_list.append(counters)

            # Update counter
            counters = counters + (torch.logical_and(self.operators.arity_one_begin<=token, token<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=token, token<self.operators.arity_two_end).long() * 2 - 1)[:, None]
            sequence_mask = torch.logical_and(sequence_mask, counters.squeeze(1) > 0)

            # Compute next parent and sibling; assemble next input tensor
            self.get_parent_sibling(n, sequences[:,1:], length-1, sequence_mask, scatter_degree, scatter_parent_where_seq, parent_child_pairs, parent_child_length, silbing_pairs, silbing_length) # update all info

        assert all_lengths.min() >= self.min_length and all_lengths.max() <= self.max_length+1 and all_lengths.max() == length
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return sequences[:,:(length+1)], all_lengths, log_probs, (scatter_degree[:,:(length+1)], all_counters_list, scatter_parent_where_seq[:,:(length+1)], parent_child_pairs[:parent_child_length[length]], parent_child_length[:length+1], silbing_pairs[:silbing_length[length]], silbing_length[:length+1])



    def get_parent_sibling(self, batch_size, sequences, recent, sequence_mask, scatter_degree, scatter_parent_where_seq, parent_child_pairs, parent_child_length, silbing_pairs, silbing_length):
        _, parent_sibling_where = self._get_parent_sibling(batch_size, sequences, recent, sequence_mask)
        parent_where, silbing_where = parent_sibling_where[:,0], parent_sibling_where[:,1]

        scatter_degree[:, recent+1] = ( scatter_degree[torch.arange(batch_size), parent_where] + \
                                 (sequences[torch.arange(batch_size), parent_where] >= self.operators.scatter_begin).long())

        scatter_parent_where_now = scatter_parent_where_seq[torch.arange(batch_size), parent_where]
        scatter_parent_where_now[sequences[:, recent]>=self.operators.scatter_begin] = recent

        scatter_parent_where_seq[:, recent+1] = scatter_parent_where_now

        parent_child_length[recent+1] = parent_child_length[recent] + len(parent_where)
        parent_child_pairs[parent_child_length[recent]: parent_child_length[recent+1], 0] = torch.arange(len(parent_where))
        parent_child_pairs[parent_child_length[recent]: parent_child_length[recent+1], 2] = recent + 2
        parent_child_pairs[parent_child_length[recent]: parent_child_length[recent+1], 1] = parent_where + 1

        where_has_sibling = torch.where(silbing_where > 0)[0]
        silbing_length[recent+1] = silbing_length[recent] + len(where_has_sibling)
        silbing_pairs[silbing_length[recent]: silbing_length[recent+1], 0] = where_has_sibling
        silbing_pairs[silbing_length[recent]: silbing_length[recent+1], 2] = recent + 2
        silbing_pairs[silbing_length[recent]: silbing_length[recent+1], 1] = silbing_where[where_has_sibling]


    def sample_sequence_train(self, sequences, info_lists):
        scatter_degree, all_counters_list, scatter_parent_where_seq, parent_child_pairs, parent_child_length, silbing_pairs, silbing_length = info_lists
        length_max = sequences.shape[1] - 1

        all_log_prob_list, all_entropy_list = [], []
        for length in range(length_max):
            output = self.forward(sequences[:,:(length+1)], scatter_degree[:,:(length+1)], parent_child_pairs[:parent_child_length[max(length-1,0)]], parent_child_pairs[parent_child_length[max(length-1,0)]:parent_child_length[length]], silbing_pairs[:silbing_length[max(length-1,0)]], silbing_pairs[silbing_length[max(length-1,0)]:silbing_length[length]])

            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, all_counters_list[length], length, sequences[:,1:(length+1)], scatter_degree=scatter_degree[:,length][:,None], scatter_parent_where_seq=scatter_parent_where_seq[:,:(length+1)])
            output = output / torch.sum(output, dim=1, keepdim=True)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            all_log_prob_list.append(dist.log_prob(sequences[:, length+1])[:, None])
            all_entropy_list.append(dist.entropy()[:, None])

        entropies = torch.cat(all_entropy_list, dim=1)
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return entropies, log_probs
