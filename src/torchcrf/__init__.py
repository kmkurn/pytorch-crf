from typing import List, Optional

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
    batch_first : bool, optional
        Whether the first dimension corresponds to the size of a minibatch.

    Attributes
    ----------
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

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduce: bool = True,
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Arguments
        ---------
        emissions : :class:`~torch.Tensor`
            Emission score tensor of size ``(seq_length, batch_size, num_tags)`` if
            ``batch_first`` is ``False``, ``(batch_size, seq_length, num_tags)`` otherwise.
        tags : :class:`~torch.LongTensor`
            Sequence of tags tensor of size ``(seq_length, batch_size)`` if
            ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        mask : :class:`~torch.ByteTensor`, optional
            Mask tensor of size ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
            ``(batch_size, seq_length)`` otherwise.
        reduce : bool, optional
            Whether to sum the log likelihood over the batch.

        Returns
        -------
        :class:`~torch.Tensor`
            The log likelihood. This will have size () if ``reduce=True``, ``(batch_size,)``
            otherwise.
        """
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if tags.dim() != 2:
            raise ValueError(f'tags must have dimension of 2, got {tags.dim()}')
        if emissions.shape[:2] != tags.shape:
            raise ValueError(
                'the first two dimensions of emissions and tags must match, '
                f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')
        if mask is not None:
            if tags.shape != mask.shape:
                raise ValueError(
                    f'size of tags and mask must match, got {tuple(tags.shape)} '
                    f'and {tuple(mask.shape)}')
            if not all(mask[0]):
                raise ValueError('mask of the first timestep must all be on')

        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        return llh if not reduce else torch.sum(llh)

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Arguments
        ---------
        emissions : :class:`~torch.Tensor`
            Emission score tensor of size ``(seq_length, batch_size, num_tags)`` if
            ``batch_first`` is ``False``, ``(batch_size, seq_length, num_tags)`` otherwise.
        mask : :class:`~torch.ByteTensor`, optional
            Mask tensor of size ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
            ``(batch_size, seq_length)`` otherwise.

        Returns
        -------
        List[List[int]]
            List of list containing the best tag sequence for each batch.
        """
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')
        if mask is not None and emissions.shape[:2] != mask.shape:
            raise ValueError(
                'the first two dimensions of emissions and mask must match, '
                f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')

        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert all(mask[0])  # TODO use .all()

        seq_length = emissions.size(0)
        mask = mask.float()

        # Start transition score
        # shape: (batch_size,)
        llh = self.start_transitions[tags[0]]

        for i in range(seq_length - 1):
            # shape: (batch_size,)
            cur_tag, next_tag = tags[i], tags[i + 1]

            # Emission score for current tag
            # shape: (batch_size,)
            # TODO use advanced indexing
            llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1) * mask[i]

            # Transition score to next tag
            # shape: (batch_size,)
            transition_score = self.transitions[cur_tag, next_tag]

            # Only add transition score if the next tag is not masked (mask == 1)
            # shape: (batch_size,)
            llh += transition_score * mask[i + 1]

        # Find last tag index
        # shape: (batch_size,)
        last_tag_indices = mask.long().sum(0) - 1
        # shape: (batch_size,)
        # TODO use advanced indexing
        last_tags = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)

        # End transition score
        # shape: (batch_size,)
        llh += self.end_transitions[last_tags]

        # Emission score for the tag in position (seq_length - 1), if mask is valid (mask == 1)
        # shape: (batch_size,)
        # TODO use advanced indexing
        # TODO check if last_tags can be replaced by tags[-1]
        llh += emissions[-1].gather(1, last_tags.view(-1, 1)).squeeze(1) * mask[-1]

        return llh

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert all(mask[0])  # TODO use .all()

        seq_length = emissions.size(0)
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size, num_tags)
        # log_prob has size of (batch_size, num_tags) where for each batch,
        # the j-th column stores the log probability that the first timestep has tag j
        log_prob = self.start_transitions.view(1, -1) + emissions[0]

        for i in range(1, seq_length):
            # Broadcast log_prob over all possible next tags
            # shape: (batch_size, num_tags, 1)
            broadcast_log_prob = log_prob.unsqueeze(2)

            # Broadcast transition score over all instances in the batch
            # shape: (1, num_tags, num_tags)
            # TODO no need to broadcast explicitly bc this is the default
            broadcast_transitions = self.transitions.unsqueeze(0)

            # Broadcast emission score over all possible current tags
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Sum current log probability, transition, and emission scores: for each sample
            # in the batch, entry in row i and column j stores the sum of scores of all
            # possible tag sequences so far, that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            score = broadcast_log_prob + broadcast_transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in log prob space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            score = torch.logsumexp(score, 1)

            # Set log_prob to the score if this timestep is valid (mask == 1), otherwise
            # leave it alone
            # shape: (batch_size, num_tags)
            log_prob = score * mask[i].unsqueeze(1) + log_prob * (1. - mask[i]).unsqueeze(1)

        # End transition score
        # shape: (batch_size, num_tags)
        log_prob += self.end_transitions.view(1, -1)

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(log_prob, 1)

    def _viterbi_decode(self, emissions: torch.FloatTensor, mask: torch.ByteTensor) \
            -> List[List[int]]:
        # TODO assert for emissions and mask
        seq_length = emissions.size(0)
        batch_size = emissions.size(1)

        # emissions: (seq_length, batch_size, num_tags)
        assert emissions.size(2) == self.num_tags

        # TODO might want to refactor this to closely mimic _compute_normalizer

        # Start transition
        viterbi_score = []
        viterbi_score.append(self.start_transitions + emissions[0])
        viterbi_path = []

        # viterbi_score is a list of tensors of shapes of (num_tags,) where value at
        # index i stores the score of the best tag sequence so far that ends with tag i
        # viterbi_path saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = viterbi_score[i - 1].view(batch_size, -1, 1)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].view(batch_size, 1, -1)

            # Compute the score matrix of shape (batch_size, num_tags, num_tags) where
            # for each sample, each entry at row i and column j stores the score of
            # transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            best_score, best_path = score.max(1)

            # Save the score and the path
            viterbi_score.append(best_score)
            viterbi_path.append(best_path)

        # Now, compute the best path for each sample
        # shape: (batch_size,)
        sequence_lengths = mask.long().sum(dim=0)
        # List to store the decoded paths
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            seq_end = sequence_lengths[idx] - 1
            _, best_last_tag = (viterbi_score[seq_end][idx] + self.end_transitions).max(0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            # TODO might wanna use seq_end here
            for path in reversed(viterbi_path[:sequence_lengths[idx] - 1]):
                best_last_tag = path[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
