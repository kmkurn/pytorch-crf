import itertools
import math
import random

from pytest import approx
import pytest
import torch
import torch.nn as nn

from torchcrf import CRF


RANDOM_SEED = 1478754


random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def compute_score(crf, emission, tag):
    assert len(emission) == len(tag)

    # Add transitions score
    score = crf.start_transitions.data[tag[0]] + crf.end_transitions.data[tag[-1]]
    for cur_tag, next_tag in zip(tag, tag[1:]):
        score += crf.transitions.data[cur_tag, next_tag]
    # Add emission score
    for emit, t in zip(emission, tag):
        score += emit[t]
    return score


def make_crf(num_tags=5):
    return CRF(num_tags)


def make_emissions(seq_length=3, batch_size=2, num_tags=5):
    return torch.autograd.Variable(torch.randn(seq_length, batch_size, num_tags),
                                   requires_grad=True)


def make_tags(seq_length=3, batch_size=2, num_tags=5):
    return torch.autograd.Variable(torch.LongTensor([
        [random.randrange(num_tags) for b in range(batch_size)]
        for _ in range(seq_length)
    ]))


class TestInit(object):
    def test_minimal(self):
        num_tags = 10
        crf = CRF(num_tags)

        assert crf.num_tags == num_tags
        assert isinstance(crf.start_transitions, nn.Parameter)
        assert crf.start_transitions.size() == (num_tags,)
        assert isinstance(crf.end_transitions, nn.Parameter)
        assert crf.end_transitions.size() == (num_tags,)
        assert isinstance(crf.transitions, nn.Parameter)
        assert crf.transitions.size() == (num_tags, num_tags)
        assert repr(crf) == f'CRF(num_tags={num_tags})'

    def test_nonpositive_num_tags(self):
        with pytest.raises(ValueError) as excinfo:
            CRF(0)
        assert 'invalid number of tags: 0' in str(excinfo.value)


class TestForward(object):
    def test_batched_loss_is_correct(self):
        crf = make_crf()
        batch_size = 10
        emissions = make_emissions(batch_size=batch_size, num_tags=crf.num_tags)
        tags = make_tags(batch_size=batch_size, num_tags=crf.num_tags)

        llh = crf(emissions, tags)

        assert isinstance(llh, torch.autograd.Variable)
        assert llh.size() == (1,)
        total_llh = 0.
        for i in range(batch_size):
            emissions_ = emissions[:, i, :].unsqueeze(1)
            tags_ = tags[:, i].unsqueeze(1)
            total_llh += crf(emissions_, tags_)

        assert llh.data[0] == approx(total_llh.data[0])

    def test_works_with_mask(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2
        emissions = make_emissions(seq_length, batch_size, crf.num_tags)
        tags = make_tags(seq_length, batch_size, crf.num_tags)
        # mask should be (seq_length, batch_size)
        mask = torch.autograd.Variable(torch.ByteTensor([
            [1, 1],
            [1, 1],
            [1, 0],
        ]))

        llh = crf(emissions, tags, mask=mask)

        # Swap seq_length and batch_size, now they're all (batch_size, seq_length, *)
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)
        # Compute manual log likelihood
        manual_llh = 0.
        for emission, tag, mask_ in zip(emissions.data, tags.data, mask.data):
            seq_len = mask_.sum()
            emission, tag = emission[:seq_len], tag[:seq_len]
            numerator = compute_score(crf, emission, tag)
            all_scores = [compute_score(crf, emission, t)
                          for t in itertools.product(range(crf.num_tags), repeat=seq_len)]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh += numerator - denominator
        # Assert equal to manual log likelihood
        assert llh.data[0] == approx(manual_llh)
        # Make sure gradients can be computed
        llh.backward()

    def test_works_without_mask(self):
        crf = make_crf()
        emissions = make_emissions(num_tags=crf.num_tags)
        tags = make_tags(num_tags=crf.num_tags)
        seq_length, batch_size = tags.size()

        llh_no_mask = crf(emissions, tags)
        # No mask means the mask is all ones
        mask = torch.autograd.Variable(torch.ones(seq_length, batch_size)).byte()
        llh_mask = crf(emissions, tags, mask=mask)

        assert llh_no_mask.data[0] == approx(llh_mask.data[0])

    def test_not_summed_over_batch(self):
        crf = make_crf()
        emissions = make_emissions(num_tags=crf.num_tags)
        tags = make_tags(num_tags=crf.num_tags)
        seq_length, batch_size = tags.size()

        llh = crf(emissions, tags, reduce=False)

        assert isinstance(llh, torch.autograd.Variable)
        assert llh.size() == (batch_size,)
        # Swap seq_length and batch_size, now they're both (batch_size, seq_length, *)
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        # Compute manual log likelihood
        manual_llh = []
        for emission, tag in zip(emissions.data, tags.data):
            numerator = compute_score(crf, emission, tag)
            all_scores = [compute_score(crf, emission, t)
                          for t in itertools.product(range(crf.num_tags), repeat=seq_length)]
            denominator = math.log(sum(math.exp(s) for s in all_scores))
            manual_llh.append(numerator - denominator)

        for llh_, manual_llh_ in zip(llh.data, manual_llh):
            assert llh_ == approx(manual_llh_)

    def test_emissions_has_bad_number_of_dimension(self):
        emissions = torch.autograd.Variable(torch.randn(1, 2), requires_grad=True)
        tags = torch.autograd.Variable(torch.LongTensor(2, 2))
        crf = make_crf()

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert 'emissions must have dimension of 3, got 2' in str(excinfo.value)

    def test_tags_has_bad_number_of_dimension(self):
        emissions = torch.autograd.Variable(torch.randn(1, 2, 3), requires_grad=True)
        tags = torch.autograd.Variable(torch.LongTensor(2, 2, 2))
        crf = make_crf(3)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert 'tags must have dimension of 2, got 3' in str(excinfo.value)

    def test_emissions_and_tags_size_mismatch(self):
        emissions = torch.autograd.Variable(torch.randn(1, 2, 3), requires_grad=True)
        tags = torch.autograd.Variable(torch.LongTensor(2, 2))
        crf = make_crf(3)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert ('the first two dimensions of emissions and tags must match, '
                'got (1, 2) and (2, 2)') in str(excinfo.value)

    def test_emissions_last_dimension_not_equal_to_number_of_tags(self):
        emissions = torch.autograd.Variable(torch.randn(1, 2, 3), requires_grad=True)
        tags = torch.autograd.Variable(torch.LongTensor(1, 2))
        crf = make_crf(10)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags)
        assert 'expected last dimension of emissions is 10, got 3' in str(excinfo.value)

    def test_mask_and_tags_size_mismatch(self):
        emissions = torch.autograd.Variable(torch.randn(1, 2, 3), requires_grad=True)
        tags = torch.autograd.Variable(torch.LongTensor(1, 2))
        mask = torch.autograd.Variable(torch.ByteTensor([[1], [1]]))
        crf = make_crf(3)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags, mask=mask)
        assert 'size of tags and mask must match, got (1, 2) and (2, 1)' in str(
            excinfo.value
        )

    def test_first_timestep_mask_is_not_all_on(self):
        emissions = torch.autograd.Variable(torch.randn(1, 2, 3), requires_grad=True)
        tags = torch.autograd.Variable(torch.LongTensor(1, 2))
        mask = torch.autograd.Variable(torch.ByteTensor([[0, 1]]))
        crf = make_crf(3)

        with pytest.raises(ValueError) as excinfo:
            crf(emissions, tags, mask=mask)
        assert 'mask of the first timestep must all be on' in str(excinfo.value)


class TestDecode(object):
    def test_batched_decode_is_correct(self):
        crf = make_crf()
        batch_size = 10
        emissions = make_emissions(batch_size=batch_size, num_tags=crf.num_tags)

        best_tags = crf.decode(emissions)

        assert len(best_tags) == batch_size
        for i in range(batch_size):
            emissions_ = emissions[:, i, :].unsqueeze(1)
            assert best_tags[i] == crf.decode(emissions_)[0]

    def test_works_without_mask(self):
        crf = make_crf()
        emissions = make_emissions(num_tags=crf.num_tags)
        seq_length = emissions.size(0)

        best_tags = crf.decode(emissions)

        # Swap seq_length and batch_size
        emissions = emissions.transpose(0, 1)
        # Compute best tag manually
        for emission, best_tag in zip(emissions.data, best_tags):
            manual_best_tag = max(itertools.product(range(crf.num_tags), repeat=seq_length),
                                  key=lambda t: compute_score(crf, emission, t))
            assert tuple(best_tag) == manual_best_tag

    def test_works_with_mask(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2
        emissions = make_emissions(seq_length, batch_size, crf.num_tags)
        # mask should be (seq_length, batch_size)
        mask = torch.autograd.Variable(torch.ByteTensor([
            [1, 1],
            [1, 1],
            [1, 0],
        ]))

        best_tags = crf.decode(emissions, mask=mask)

        # Swap seq_length and batch_size, now they're all (batch_size, seq_length, *)
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)
        # Compute best tag manually
        for emission, best_tag, mask_ in zip(emissions.data, best_tags, mask.data):
            seq_len = mask_.sum()
            assert len(best_tag) == seq_len
            emission = emission[:seq_len]
            manual_best_tag = max(itertools.product(range(crf.num_tags), repeat=seq_len),
                                  key=lambda t: compute_score(crf, emission, t))
            assert tuple(best_tag) == manual_best_tag

    def test_works_with_tensor(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2
        emissions_var = make_emissions(seq_length, batch_size, crf.num_tags)
        emissions = emissions_var.data
        # mask should be (seq_length, batch_size)
        mask = torch.ByteTensor([
            [1, 1],
            [1, 1],
            [1, 0],
        ])
        mask_var = torch.autograd.Variable(mask)

        best_tags = crf.decode(emissions, mask=mask)
        best_tags_var = crf.decode(emissions_var, mask=mask_var)

        assert best_tags == best_tags_var

    def test_emissions_has_bad_number_of_dimension(self):
        emissions = torch.autograd.Variable(torch.randn(1, 2))
        crf = make_crf()

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions)
        assert 'emissions must have dimension of 3, got 2' in str(excinfo.value)

    def test_emissions_last_dimension_not_equal_to_number_of_tags(self):
        emissions = torch.autograd.Variable(torch.randn(1, 2, 3))
        crf = make_crf(10)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions)
        assert 'expected last dimension of emissions is 10, got 3' in str(excinfo.value)

    def test_emissions_and_mask_size_mismatch(self):
        emissions = torch.autograd.Variable(torch.randn(1, 2, 3))
        mask = torch.autograd.Variable(torch.ByteTensor([[1, 1], [1, 0]]))
        crf = make_crf(3)

        with pytest.raises(ValueError) as excinfo:
            crf.decode(emissions, mask=mask)
        assert ('the first two dimensions of emissions and mask must match, '
                'got (1, 2) and (2, 2)') in str(excinfo.value)

    def test_batched_decode(self):
        batch_size, seq_len, num_tags = 2, 3, 4
        crf = CRF(num_tags)
        emissions = torch.randn(seq_len, batch_size, num_tags)
        mask = torch.ByteTensor([[1, 1, 1], [1, 1, 0]]).transpose(0, 1)

        # non-batched
        non_batched = []
        for emissions_, mask_ in zip(emissions.transpose(0, 1), mask.transpose(0, 1)):
            emissions_ = emissions_.unsqueeze(1)  # shape: (seq_len, 1, num_tags)
            mask_ = mask_.unsqueeze(1)  # shape: (seq_len, 1)
            result = crf.decode(emissions_, mask=mask_)
            assert len(result) == 1
            non_batched.append(result[0])

        # batched
        batched = crf.decode(emissions, mask=mask)

        assert non_batched == batched

    def test_marginal_prob(self):
        crf = make_crf()
        seq_length, batch_size = 3, 2
        emissions = make_emissions(seq_length, batch_size, crf.num_tags)
        tags = make_tags(seq_length, batch_size, crf.num_tags)
        # mask should be (seq_length, batch_size)
        mask = torch.autograd.Variable(torch.ByteTensor([
            [1, 1],
            [1, 1],
            [1, 0],
        ]))

        probs = crf.compute_log_marginal_probabilities(emissions, mask=mask)

        assert probs.shape == emissions.shape