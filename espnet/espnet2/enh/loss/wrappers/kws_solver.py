from collections import defaultdict
from itertools import permutations

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.enh.loss.wrappers.pit_solver import PITSolver
from espnet2.enh.loss.wrappers.fixed_order import FixedOrderSolver

class KWSSolver(AbsLossWrapper):
    def __init__(
        self,
        criterion: AbsEnhLoss,
        weight=1.0,
        independent_perm=True,
        flexible_numspk=False,
    ):
        """Permutation Invariant Training Solver
        Args:
            criterion (AbsEnhLoss): an instance of AbsEnhLoss
            weight (float): weight (between 0 and 1) of current loss
                for multi-task learning.
            independent_perm (bool):
                If True, PIT will be performed in forward to find the best permutation;
                If False, the permutation from the last LossWrapper output will be
                inherited.
                NOTE (wangyou): You should be careful about the ordering of loss
                    wrappers defined in the yaml config, if this argument is False.
            flexible_numspk (bool):
                If True, num_spk will be taken from inf to handle flexible numbers of
                speakers. This is because ref may include dummy data in this case.
        """
        super().__init__()
        self.criterion = criterion
        self.weight = weight
        self.independent_perm = independent_perm
        self.flexible_numspk = flexible_numspk
        self.fix = FixedOrderSolver(criterion,weight)
        self.pit = PITSolver(criterion,weight,independent_perm,flexible_numspk)

    def forward(self, ref, inf, others={}):
        """PITSolver forward.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]

        Returns:
            loss: (torch.Tensor): minimum loss with the best permutation
            stats: dict, for collecting training status
            others: dict, in this PIT solver, permutation order will be returned
        """
        # import pdb; pdb.set_trace()
        wkp_ind = others['wkp_ind']
        stats = defaultdict(list)
        ref_pit = tuple(r[wkp_ind==0]for r in ref)
        inf_pit = tuple(r[wkp_ind==0]for r in inf)
        
        ref_wkp = tuple(r[wkp_ind==1]for r in ref)
        inf_wkp = tuple(r[wkp_ind==1]for r in inf)

        loss, _, _ = self.pit(ref, inf)

        if ref_wkp[0].size(0) > 0:
            l_wkp, s_wkp, _ = self.fix(ref_wkp,inf_wkp)
            loss += l_wkp

        # if ref_wkp[0].size(0) > 0 and ref_pit[0].size(0):
        #     l_wkp, s_wkp, _ = self.fix(ref_wkp,inf_wkp)
        #     l_pit, s_pit, _ = self.pit(ref_pit,inf_pit)
        #     loss = (l_wkp * ref_wkp[0].size(0) + l_pit * ref_pit[0].size(0)) / ref[0].size(0)
        # elif ref_wkp[0].size(0) > 0:
        #     l_wkp, s_wkp, _ = self.fix(ref_wkp,inf_wkp)
        #     loss = l_wkp
        # else:
        #     l_pit, s_pit, _ = self.pit(ref_pit,inf_pit)
        #     loss = l_pit
        stats[self.criterion.name] = loss.detach()
        # pass
        # perm没用上那我就不回传了
        return loss.mean(), dict(stats), {}