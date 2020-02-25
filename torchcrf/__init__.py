__version__ = '0.7.2'

from typing import Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
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
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if emissions.shape[:2] != tags.shape:
            raise ValueError(
                'the first two dimensions of emissions and tags must match, '
                f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        self._validate(emissions, mask)

        seq_length, batch_size = tags.shape
        # Start transition score and first emission for both
        # shape: (batch_size,)
        numerator = self.start_transitions[tags[0]]
        numerator += emissions[0, torch.arange(batch_size), tags[0]]
        denominator = self.start_transitions + emissions[0]

        # Broadcast emissions
        # shape: (seq_length, batch_size, 1, num_tags)
        broadcast_emissions = emissions.unsqueeze(2)
        for i in range(1, seq_length):
            # Add transition and emission to next tag to the numerator if
            # the next timestep is valid (mask==1)
            # shape: (batch_size,)
            numerator += (
                self.transitions[tags[i - 1], tags[i]] +
                emissions[i, torch.arange(batch_size), tags[i]]) * mask[i].float()

            # Broadcast the denominator for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_denominator = denominator.unsqueeze(2)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_denominator = broadcast_denominator + self.transitions + broadcast_emissions[i]

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_denominator = torch.logsumexp(next_denominator, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            denominator = torch.where(mask[i].unsqueeze(1), next_denominator, denominator)

        # End transition score for numerator
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        numerator += self.end_transitions[last_tags]

        # End transition score for denominator
        # shape: (batch_size, num_tags)
        denominator += self.end_transitions
        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        denominator = torch.logsumexp(denominator, dim=1)

        # Compute log likelihood
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        elif reduction == 'sum':
            return llh.sum()
        elif reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def _validate(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')
        if emissions.shape[:2] != mask.shape:
            raise ValueError(
                'the first two dimensions of emissions and mask must match, '
                f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
        if not mask[0].all():
            raise ValueError('mask of the first timestep must all be on')
        if not ((torch.abs(mask[:-1].long() - mask[1:].long()).sum(dim=0)) <= 1).all():
            raise ValueError('mask must not be discontinuous')

    def decode(
            self, emissions: torch.Tensor,
            mask: Optional[torch.ByteTensor] = None) -> torch.LongTensor:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            `~torch.LongTensor`: Tensor of size size ``(batch_size, seq_length)`` 
            containing the most likely tags sequences. If a mask was specified,
            the associated tags of each sequence are random and should be 
            discarded.
        """
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        self._validate(emissions, mask)

        seq_length, batch_size = mask.shape

        # score is a tensor where for every batch, value at column j stores the score of the
        # best tag sequence so far that ends # with tag j
        # history saves where the best tags candidate transitioned from

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        indices = emissions.new_empty(batch_size, self.num_tags, dtype=torch.long)
        history = emissions.new_empty(
            seq_length - 1, batch_size, self.num_tags, dtype=torch.long)

        # Broadcast emission score for every possible current tag
        # shape: (seq_length, batch_size, 1, num_tags)
        broadcast_emissions = emissions.unsqueeze(2)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions[i]

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, next_indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            indices = torch.where(mask[i - 1].unsqueeze(1), next_indices, indices)
            history[-i] = indices

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best tag sequences for each sample
        # shape: (batch_size, seq_length)
        best_tags = emissions.new_empty(batch_size, seq_length, dtype=torch.long)

        # Find the tag which maximizes the score at the last timestep
        # shape: (batch_size, )
        _, best_last_tag = score.max(dim=1)
        best_prev_tag = best_last_tag

        # Trace back where the best last tag comes from, and add to our sequences
        for i in range(seq_length - 1):
            best_prev_tag = history[i, torch.arange(batch_size), best_prev_tag]
            best_tags[:, seq_length - 2 - i] = best_prev_tag
        # Add the tag for the last timestep
        best_tags[torch.arange(batch_size), mask.long().sum(dim=0) - 1] = best_last_tag

        return best_tags
