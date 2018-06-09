from typing import List, Optional, Union

from torch.autograd import Variable
import torch
import torch.nn as nn

class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP]. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has ``decode`` method which finds the
    best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Arguments
    ---------
    num_tags : int
        Number of tags.

    Attributes
    ----------
    num_tags : int
        Number of tags passed to ``__init__``.
    start_transitions : :class:`~torch.nn.Parameter`
        Start transition score tensor of size ``(num_tags,)``.
    end_transitions : :class:`~torch.nn.Parameter`
        End transition score tensor of size ``(num_tags,)``.
    transitions : :class:`~torch.nn.Parameter`
        Transition score tensor of size ``(num_tags, num_tags)``.

    References
    ----------
    .. [LMP] Lafferty, J., McCallum, A., Pereira, F. (2001).
             "Conditional random fields: Probabilistic models for segmenting and
             labeling sequence data". *Proc. 18th International Conf. on Machine
             Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """
    def __init__(self, num_tags: int) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.Tensor(num_tags))
        self.end_transitions = nn.Parameter(torch.Tensor(num_tags))
        self.transitions = nn.Parameter(torch.Tensor(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform(self.start_transitions, -0.1, 0.1)
        nn.init.uniform(self.end_transitions, -0.1, 0.1)
        nn.init.uniform(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self,
                emissions: Variable,
                tags: Variable,
                mask: Optional[Variable] = None,
                reduce: bool = True,
                ) -> Variable:
        """Compute the log likelihood of the given sequence of tags and emission score.

        Arguments
        ---------
        emissions : :class:`~torch.autograd.Variable`
            Emission score tensor of size ``(seq_length, batch_size, num_tags)``.
        tags : :class:`~torch.autograd.Variable`
            Sequence of tags as ``LongTensor`` of size ``(seq_length, batch_size)``.
        mask : :class:`~torch.autograd.Variable`, optional
            Mask tensor as ``ByteTensor`` of size ``(seq_length, batch_size)``.
        reduce : bool
            Whether to sum the log likelihood over the batch.

        Returns
        -------
        :class:`~torch.autograd.Variable`
            The log likelihood. This will have size (1,) if ``reduce=True``, ``(batch_size,)``
            otherwise.
        """
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if tags.dim() != 2:
            raise ValueError(f'tags must have dimension of 2, got {tags.dim()}')
        if emissions.size()[:2] != tags.size():
            raise ValueError(
                'the first two dimensions of emissions and tags must match, '
                f'got {tuple(emissions.size()[:2])} and {tuple(tags.size())}'
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}'
            )
        if mask is not None:
            if tags.size() != mask.size():
                raise ValueError(
                    f'size of tags and mask must match, got {tuple(tags.size())} '
                    f'and {tuple(mask.size())}'
                )
            if not all(mask[0].data):
                raise ValueError('mask of the first timestep must all be on')

        if mask is None:
            mask = Variable(self._new(tags.size()).fill_(1)).byte()

        numerator = self._compute_joint_llh(emissions, tags, mask)
        denominator = self._compute_log_partition_function(emissions, mask)
        llh = numerator - denominator
        return llh if not reduce else torch.sum(llh)

    def decode(self,
               emissions: Union[Variable, torch.FloatTensor],
               mask: Optional[Union[Variable, torch.ByteTensor]] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Arguments
        ---------
        emissions : :class:`~torch.autograd.Variable` or :class:`~torch.FloatTensor`
            Emission score tensor of size ``(seq_length, batch_size, num_tags)``.
        mask : :class:`~torch.autograd.Variable` or :class:`torch.ByteTensor`
            Mask tensor of size ``(seq_length, batch_size)``.

        Returns
        -------
        list
            List of list containing the best tag sequence for each batch.
        """
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}'
            )
        if mask is not None and emissions.size()[:2] != mask.size():
            raise ValueError(
                'the first two dimensions of emissions and mask must match, '
                f'got {tuple(emissions.size()[:2])} and {tuple(mask.size())}'
            )

        if isinstance(emissions, Variable):
            emissions = emissions.data
        if mask is None:
            mask = self._new(emissions.size()[:2]).fill_(1).byte()
        elif isinstance(mask, Variable):
            mask = mask.data

        return self._viterbi_decode(emissions, mask)

    def _compute_joint_llh(self,
                           emissions: Variable,
                           tags: Variable,
                           mask: Variable) -> Variable:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.size()[:2] == tags.size()
        assert emissions.size(2) == self.num_tags
        assert mask.size() == tags.size()
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()

        # Start transition score
        llh = self.start_transitions[tags[0]]  # (batch_size,)

        for i in range(seq_length - 1):
            cur_tag, next_tag = tags[i], tags[i+1]
            # Emission score for current tag
            llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1) * mask[i]
            # Transition score to next tag
            transition_score = self.transitions[cur_tag, next_tag]
            # Only add transition score if the next tag is not masked (mask == 1)
            llh += transition_score * mask[i+1]

        # Find last tag index
        last_tag_indices = mask.long().sum(0) - 1  # (batch_size,)
        last_tags = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)

        # End transition score
        llh += self.end_transitions[last_tags]
        # Emission score for the last tag, if mask is valid (mask == 1)
        llh += emissions[-1].gather(1, last_tags.view(-1, 1)).squeeze(1) * mask[-1]

        return llh

    def _compute_log_alpha(self,
                           emissions: Variable,
                           mask: Variable) -> Variable:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.size()[:2] == mask.size()
        assert emissions.size(2) == self.num_tags
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()

        # Start transition score and first emission
        log_prob = [self.start_transitions.view(1, -1) + emissions[0]]
        # Here, log_prob has size (batch_size, num_tags) where for each batch,
        # the j-th column stores the log probability that the current timestep has tag j

        for i in range(1, seq_length):
            # Broadcast log_prob over all possible next tags
            broadcast_log_prob = log_prob[i-1].unsqueeze(2)  # (batch_size, num_tags, 1)
            # Broadcast transition score over all instances in the batch
            broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            # Broadcast emission score over all possible current tags
            broadcast_emissions = emissions[i].unsqueeze(1)  # (batch_size, 1, num_tags)
            # Sum current log probability, transition, and emission scores
            score = broadcast_log_prob + broadcast_transitions \
                + broadcast_emissions  # (batch_size, num_tags, num_tags)
            # Sum over all possible current tags, but we're in log prob space, so a sum
            # becomes a log-sum-exp
            score = self._log_sum_exp(score, 1)  # (batch_size, num_tags)
            # Set log_prob to the score if this timestep is valid (mask == 1), otherwise
            # leave it alone
            log_prob.append(score * mask[i].unsqueeze(1) + log_prob[i-1] * (1.-mask[i]).unsqueeze(1))

        # End transition score
        log_prob[len(log_prob)-1] += self.end_transitions.view(1, -1)
        return torch.stack(log_prob)

    def _compute_log_partition_function(self,
                                        emissions: Variable,
                                        mask: Variable) -> Variable:
        alpha = self._compute_log_alpha(emissions, mask)
        # Sum (log-sum-exp) over all possible tags at the last time stamp
        return self._log_sum_exp(alpha[alpha.size(0)-1], 1)  # (batch_size,)

    def _compute_log_beta(self,
                          emissions: Variable,
                          mask: Variable) -> Variable:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.size()[:2] == mask.size()
        assert emissions.size(2) == self.num_tags
        assert all(mask[0].data)

        seq_length = emissions.size(0)
        mask = mask.float()

        # End transition score and last emission
        log_prob = [self.end_transitions.view(1, -1) + emissions[seq_length-1]]
        # Here, log_prob has size (batch_size, num_tags) where for each batch,
        # the j-th column stores the log probability that the current timestep has tag j

        for i in range(1, seq_length):
            # Broadcast log_prob over all possible next tags
            broadcast_log_prob = log_prob[i-1].unsqueeze(2)  # (batch_size, num_tags, 1)
            # Broadcast transition score over all instances in the batch
            broadcast_transitions = self.transitions.transpose(0,1).unsqueeze(0)  # (1, num_tags, num_tags)
            # Broadcast emission score over all possible current tags
            broadcast_emissions = emissions[seq_length-i-1].unsqueeze(1)  # (batch_size, 1, num_tags)
            # Sum current log probability, transition, and emission scores
            score = broadcast_log_prob + broadcast_transitions \
                + broadcast_emissions  # (batch_size, num_tags, num_tags)
            # Sum over all possible current tags, but we're in log prob space, so a sum
            # becomes a log-sum-exp
            score = self._log_sum_exp(score, 1)  # (batch_size, num_tags)
            # Set log_prob to the score if this timestep is valid (mask == 1), otherwise
            # leave it alone
            log_prob.append(score * mask[seq_length-i-1].unsqueeze(1) + log_prob[i-1] * (1.-mask[seq_length-i-1]).unsqueeze(1))

        # End transition score
        log_prob[len(log_prob)-1] += self.start_transitions.view(1, -1)

        log_prob.reverse()

        return torch.stack(log_prob)

    def compute_log_marginal_probabilities(self, emissions: Variable, mask: Variable) -> Variable:
        alpha = self._compute_log_alpha(emissions,mask)
        beta = self._compute_log_beta(emissions,mask)
        z = self._log_sum_exp(alpha[alpha.size(0)-1], 1)

        prob = alpha + beta - z.view(1,-1,1)
        s = torch.nn.Softmax(dim=2)
        return s(prob)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor) \
            -> List[List[int]]:
        # Get input sizes
        max_sequence_length = emissions.shape[0]
        minibatch_size = emissions.shape[1]
        sequence_lengths = mask.long().sum(dim=0)

        # emissions: (seq_length, batch_size, num_tags)
        assert emissions.shape[2] == self.num_tags

        # list to store the decoded paths
        best_tags_list = []

        # Start transition
        viterbi_score = []
        viterbi_score.append(self.start_transitions.data + emissions[0])
        viterbi_path = []

        # Here, viterbi_score is a list of shapes of (num_tags,) where value at index i stores
        # the score of the best tag sequence so far that ends with tag i
        # viterbi_path saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, max_sequence_length):
            # Broadcast viterbi score for every possible next tag
            broadcast_score = viterbi_score[i - 1].view(minibatch_size, -1, 1)
            # Broadcast emission score for every possible current tag
            broadcast_emission = emissions[i].view(minibatch_size, 1, -1)
            # Compute the score matrix of shape (num_tags, num_tags) where each entry at
            # row i and column j stores the score of transitioning from tag i to tag j
            # and emitting
            score = broadcast_score + self.transitions.data + broadcast_emission
            # Find the maximum score over all possible current tag
            best_score, best_path = score.max(1)  # (minibatch_size,num_tags,)
            # Save the score and the path
            viterbi_score.append(best_score)
            viterbi_path.append(best_path)

        # Now, compute the best path for each sample
        for idx in range(minibatch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            seq_end = sequence_lengths[idx]-1
            _, best_last_tag = (viterbi_score[seq_end][idx] + self.end_transitions.data).max(0)
            best_tags = [best_last_tag[0]]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for path in reversed(viterbi_path[:sequence_lengths[idx] - 1]):
                best_last_tag = path[idx][best_tags[-1]]
                best_tags.append(best_last_tag)

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list

    @staticmethod
    def _log_sum_exp(tensor: Variable, dim: int) -> Variable:
        # Find the max value along `dim`
        offset, _ = tensor.max(dim)
        # Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)
        # Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))
        # Add offset back
        return offset + safe_log_sum_exp

    def _new(self, *args, **kwargs) -> torch.FloatTensor:
        param = next(self.parameters())
        return param.data.new(*args, **kwargs)
