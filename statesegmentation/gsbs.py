from numpy import cov, ndarray, nonzero, ones, triu, unique, zeros, arange, max, where, sum, int, copy, unravel_index, \
    argsort, cumsum, all, equal, array
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_ind
from typing import Optional
from tqdm import tqdm
import warnings


class GSBS:
    def __init__(self, kmax: int, x: ndarray, statewise_detection: Optional[bool] = True, finetune: Optional[int] = 1,
                 finetune_order: Optional[bool] = True, y: Optional[ndarray] = None, blocksize: Optional[int] = 50,
                 dmin: Optional[int] = 1) -> None:
        """Given an ROI timeseries, this class uses a greedy search algorithm (GSBS) to segment the timeseries into
        neural states with stable activity patterns. GSBS identifies the timepoints of neural state transitions,
        while t-distance is used to determine the optimal number of neural states.
        You can find more information about the method here:
        Geerligs L., van Gerven M., Güçlü U. (2021) Detecting neural state transitions underlying event segmentation
        Neuroimage. https://doi.org/10.1016/j.neuroimage.2021.118085
        The method has since been improved as described here:
        Geerligs L., Gözükara D., Oetringer D., Campbell K., van Gerven M., Güçlü U (2022)
        A partially nested cortical hierarchy of neural states underlies event segmentation in the human brain
        BioRxiv. https://doi.org/10.1101/2021.02.05.429165

        Arguments:
            kmax {int} -- the maximal number of neural states that should be estimated in the greedy search. To estimate
                          the optimal number of states, half the number of timepoints is a reasonable choice for kmax.
            x {ndarray} -- a multivoxel ROI timeseries - timepoint by voxels array
        Keyword Arguments:
            statewise_detection {Optional[bool]} -- if True, a 2-D search for optimal boundary locations is performed.
                                                    This allows the algorithm to determine the location of a new state,
                                                    rather than identifying a boundary between two states.
                                                    A restriction to the search is that both boundaries must be placed
                                                    within a single previously existing state. The algorithm will place
                                                    one or two boundaries at each iteration, based on which of these
                                                    options results in the highest t-distance. Therefore, it is possible
                                                    that the algorithm returns kmax+1 states. This improvement in GSBS is
                                                    despribed in more detail in the BioRxiv paper.
                                                    If False, the original GSBS implementation as formulated in the
                                                    Neuroimage paper is used, which adds one boundary per iteration.
                                                    (default: {True})
            finetune {Optional[int]} --    the number of TRs around each state boundary in which the algorithm searches
                                            for the optimal boundary during the finetuning step. If finetune is 0,
                                            no finetuning of state boundaries is performed. If finetune is <0 all TRs
                                            are included during the finetuning step. Note that the latter option will be
                                            computationally intensive. (default: {1})
            finetune_order  {Optional[int]} -- the order in which boundaries are finetuned is determined by the boundary
                                               strength. If True, the finetuning of boundaries is performed in the order
                                               of weakest-strongest boundary. If False, the funetuning is perfomed in the
                                               order of strongest-weakest. (default: {True})
                                               Note that this is different from the original implementation in the
                                               Neuroimage paper where the order of finetuning was based on the order
                                               in which boundaries were detected.
            y {Optional[ndarray]} -- a multivoxel ROI timeseries - timepoint by voxels array
                                      if y is given, the t-distance will be based on cross-validation,
                                      such that the state boundaries are identified using the data in x and the
                                      optimal number of states is identified using the data in y. If y is not given
                                      the state boundaries and optimal number of states are both based on x.
                                      (default: {None})
            blocksize {Optional[int]} -- to speed up the computation when the number of timepoints is large, the algorithm
                                        can first detect local optima for boundary locations within a block of one or
                                        multiple states before obtaining the optimum across all states. Blocksize
                                        indicates the minimal number of timepoints that constitute a block. (default: {50})
            dmin {Optional[int]} -- the number of TRs around the diagonal of the time by time correlation matrix that
                                    are not taken into account in the computation of the t-distance metric. For the
                                    default value of 1, only the diagonal itself is not taken into account (default: {1})
        """

        self.kmax = kmax
        self.x = x
        self.statewise_detection = statewise_detection
        self.finetune = finetune
        self.finetune_order = finetune_order
        self.y = y
        self.blocksize = blocksize
        self.dmin = dmin

        self._argmax = None
        self.all_bounds = zeros((self.kmax + 2, self.x.shape[0]), int)
        self._bounds = zeros(self.x.shape[0], int)
        self._deltas = zeros(self.x.shape[0], bool)
        self._tdists = zeros(self.kmax + 2, float)

    def get_bounds(self, k: int = None) -> ndarray:
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the boundaries for the optimal number of states (k=nstates).
                When k is given, the boundaries for k states are returned. If self.statewise_detection is true,
                the number of states in the returned ndarray can be k+1 rather than k.
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a higher number indicates a state transition. State transitions
            are numbered in the order in which they are detected in GSBS (stronger boundaries tend
            to be detected first).
            If self.statewise_detection is true, two boundaries may be added at the same time,
            in which case two timepoints will have the value X (depending on when the boundary was added) while no
            timepoints will have the value X-1.

        """
        assert self._argmax is not None
        if k is None:
            k = self._argmax

        return self.all_bounds[k]

    @property
    def bounds(self) -> ndarray:
        """
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a higher number indicates a state transition. State transitions
            are numbered in the order in which they are detected in GSBS (stronger boundaries tend
            to be detected first). If self.statewise_detection is true, two boundaries may be added at the same time,
            in which case two timepoints will have the value X (depending on when the boundary was added) while no
            timepoints will have the value X-1.
            The number of states is equal to the optimal number of states (nstates).
        """
        return self.get_bounds(k=None)

    def get_deltas(self, k: int = None) -> ndarray:
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the deltas for the optimal number of states (k=nstates).
                When k is given, the deltas for k states are returned.
                If self.statewise_detection is true, the number of states in the returned ndarray can be k+1 rather than k.
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a one indicates a state transition.

        """
        assert self._argmax is not None

        if k is None:
            k = self._tdists.argmax()

        bounds = self.all_bounds[k]
        deltas = bounds > 0
        deltas = deltas * 1

        return deltas

    @property
    def deltas(self) -> ndarray:
        """
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a one indicates a state transition.
            The number of states is equal to the optimal number of states (nstates).
        """
        return self.get_deltas(k=None)

    @property
    def tdists(self) -> ndarray:
        """
        Returns:
            ndarray -- array with length == kmax
            contains the t-distance estimates for each value of k (number of states)
        """
        assert self._argmax is not None
        return self._tdists

    def get_states(self, k: int = None) -> ndarray:
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the states for the optimal number of states (k=nstates).
                When k is given, k states are returned.
                If self.statewise_detection is true, the number of states in the returned ndarray can be k+1 rather than k.
        Returns:
            ndarray -- array with length == number of timepoints,
            where each timepoint is numbered according to the neural state it is in.
        """
        assert self._argmax is not None
        if k is None:
            k = self._argmax
        states = self._states(self.get_deltas(k))
        return states

    @property
    def states(self) -> ndarray:
        """
        Returns:
            ndarray -- array with length == number of timepoints, where each timepoint is numbered according to
            the neural state it is in. The number of states is equal to the optimal number of states (nstates).
        """
        return self.get_states(k=None)

    @property
    def nstates(self) -> ndarray:
        """
        Returns:
            integer -- optimal number of states as determined by t-distance
        """
        assert self._argmax is not None
        return self._argmax

    def get_state_patterns(self, k: int = None) -> ndarray:
        """
        Keyword Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the state patterns for the optimal number of states (k=nstates).
                When k is given, the state patterns for k states are returned.
                If self.statewise_detection is true, the number of states in the returned ndarray can be k+1 rather than k.
        Returns:
            ndarray -- timepoint by nstates array
            Contains the average voxel activity patterns for each of the estimates neural states
        """
        assert self._argmax is not None
        if k is None:
            k = self._argmax
        deltas = self.get_deltas(k)
        states = self._states(deltas)
        states_unique = unique(states)
        xmeans = zeros((len(states_unique), self.x.shape[1]), float)

        for state_idx, state in enumerate(states_unique):
            xmeans[state_idx] = self.x[state == states].mean(0)

        return xmeans

    @property
    def state_patterns(self) -> ndarray:
        """
        Returns:
            ndarray -- timepoint by nstates array
            Contains the average voxel activity patterns for each of the estimates neural states.
            The number of states is equal to the optimal number of states (nstates).
        """
        return self.get_state_patterns(k=None)

    def get_strengths(self, k=None) -> ndarray:
        """
        Arguments:
            k {Optional[int]} -- number of states
                By default the function returns the state patterns for the optimal number of states (k=nstates).
                When k is given, the state patterns for k states are returned.
                If self.statewise_detection is true, the number of states in the returned ndarray can be k+1 rather than k.
        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and another value indicates a state transition. The numbers indicate
            the strength of a state transition, as defined by the Pearson correlation-distance between neural
            activity patterns in consecutive states.
        """
        if k is None:
            assert self._argmax is not None
            k = self._argmax

        deltas = self.all_bounds[k] > 0
        states = self._states(deltas)

        states_unique = unique(states)
        pcorrs = zeros(len(states_unique) - 1, float)
        xmeans = zeros((len(states_unique), self.x.shape[1]), float)

        for state_idx, state in enumerate(states_unique):
            xmeans[state_idx] = self.x[state == states].mean(0)
            if state_idx > 0:
                pcorrs[state_idx - 1] = pearsonr(xmeans[state_idx], xmeans[state_idx - 1])[0]

        strengths = zeros(deltas.shape, float)
        strengths[deltas == 1] = 1 - pcorrs

        return strengths

    @property
    def strengths(self) -> ndarray:
        """
         Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and another value indicates a state transition. The numbers indicate
            the strength of a state transition, as defined by the correlation-distance between neural
            activity patterns in consecutive states. The number of states is equal to the optimal number of states (nstates).
        """

        return self.get_strengths()

    def fit(self, showProgressBar=True) -> None:

        """This function performs the GSBS and t-distance computations to determine
        the location of state boundaries and the optimal number of states.
        """

        if self._argmax is not None:
            warnings.warn("The algorithm has already been performed. Returning.")
            return

        ind = triu(ones(self.x.shape[0], bool), self.dmin)
        z = GSBS._zscore(self.x)

        if self.y is None:
            t = cov(z)[ind]
        else:
            t = cov(GSBS._zscore(self.y))[ind]

        x = self.x
        z = z

        k = 2
        with tqdm(total=self.kmax - 1, disable=not showProgressBar) as pbar:
            while k < self.kmax + 1:
                states = self._states(self._deltas)
                wdists, wdists_s = self._wdists_blocks(self._deltas, states, x, z, self.statewise_detection,
                                                       blocksize=self.blocksize)
                increment = 0

                argmax = wdists.argmax()
                deltas = copy(self._deltas)
                deltas[argmax] = True
                tdist = GSBS._tdist(deltas, t, ind)

                if self.statewise_detection and wdists_s is not None:
                    argmax_s = unravel_index(wdists_s.argmax(), (x.shape[0], x.shape[0]))
                    deltas_s = copy(self._deltas)
                    deltas_s[argmax_s[0]] = True
                    deltas_s[argmax_s[1]] = True
                    tdist_s = GSBS._tdist(deltas_s, t, ind)
                    if tdist_s > tdist:
                        self._deltas = copy(deltas_s)
                        self._bounds[argmax_s[0]] = k + 1
                        self._bounds[argmax_s[1]] = k + 1
                        increment = 2
                if not self.statewise_detection or increment == 0:
                    self._deltas = copy(deltas)
                    self._bounds[argmax] = k
                    increment = 1

                self.all_bounds[k:k + increment] = self._bounds
                if self.finetune != 0 and k > 2:
                    self._bounds = self._finetune(self, self._bounds, x, z, self.finetune, self.finetune_order)
                    self._deltas = self._bounds > 0
                    self.all_bounds[k:k + increment] = self._bounds

                if increment > 1:
                    self._tdists[k] = self._tdists[k - 1]
                    self._tdists[k + 1] = GSBS._tdist(self._deltas, t, ind)
                else:
                    self._tdists[k] = GSBS._tdist(self._deltas, t, ind)

                k = k + increment
                pbar.update(increment)

        self._argmax = self._tdists.argmax()

    @staticmethod
    def _finetune(self, bounds: ndarray, x: ndarray, z: ndarray, finetune: int, finetune_order: bool):

        finebounds = copy(bounds.astype(int))
        strengths = self.get_strengths(sum(bounds > 0) + 1)
        if finetune_order:
            order = argsort(strengths)
        else:
            order = argsort(-strengths)
        sorted_strength = strengths[order]

        nonzero = where(sorted_strength != 0)[0]
        for ind in order[nonzero]:
            state_id = finebounds[ind]
            finebounds[ind] = 0

            deltas = finebounds > 0
            states = self._states(deltas)

            if finetune < 0:
                boundopt = arange(1, states.shape[0], 1)
            else:
                boundopt = arange(max((1, ind - finetune)), min((states.shape[0], ind + finetune + 1)), 1, dtype=object)
            wdists = self._wdists(deltas, states, x, z, boundopt)

            argmax = wdists.argmax()
            finebounds[argmax] = state_id

        return finebounds

    @staticmethod
    def _states(deltas: ndarray) -> ndarray:
        return cumsum(deltas) + 1

    @staticmethod
    def _tdist(deltas: ndarray, t: ndarray, ind) -> ndarray:
        states = GSBS._states(deltas)[:, None]
        c_diff, same, alldiff = (lambda c: (c == 1, c == 0, c > 0))(cdist(states, states, "cityblock")[ind])
        tdist = 0 if sum(same) < 2 else ttest_ind(t[same], t[c_diff], equal_var=False)[0]

        return tdist

    @staticmethod
    def _wdists(deltas: ndarray, states: ndarray, x: ndarray, z: ndarray, boundopt: ndarray = None) -> ndarray:
        xmeans = zeros(x.shape, float)
        wdists = -ones(x.shape[0], float)

        if boundopt is None:
            boundopt = arange(1, x.shape[0])

        for state in map(lambda s: s == states, unique(states)):
            xmeans[state] = x[state].mean(0)

        for i in boundopt:
            if deltas[i] == 0:
                state = nonzero(states[i] == states)[0]
                xmean = copy(xmeans[state])
                xmeans[state[0]: i] = x[state[0]: i].mean(0)
                xmeans[i: state[-1] + 1] = x[i: state[-1] + 1].mean(0)
                wdists[i] = xmeans.shape[1] * (GSBS._zscore(xmeans) * z).mean() / (xmeans.shape[1] - 1)
                xmeans[state] = xmean

        return wdists

    @staticmethod
    def _wdists_state(deltas: ndarray, states: ndarray, x: ndarray, z: ndarray, stateopt: ndarray = None) -> ndarray:
        xmeans = zeros(x.shape, float)
        wdists = -ones((x.shape[0], x.shape[0]), float)

        for state in map(lambda s: s == states, unique(states)):
            xmeans[state] = x[state].mean(0)

        for i in arange(1, x.shape[0]):
            if deltas[i] == 0:
                if not stateopt is None and max(sum(equal(stateopt, array([[i, 0]])))) < 1:
                    continue
                state = nonzero(states[i] == states)[0]
                for j in state:
                    if j > i:
                        if not stateopt is None and max(sum(equal(stateopt, array([[i, j]])))) < 2:
                            continue
                        xmean = copy(xmeans[state])
                        xmeans[state[0]: i] = x[state[0]: i].mean(0)
                        xmeans[i: j] = x[i: j].mean(0)
                        xmeans[j: state[-1] + 1] = x[j: state[-1] + 1].mean(0)
                        wdists[i, j] = xmeans.shape[1] * (GSBS._zscore(xmeans) * z).mean() / (xmeans.shape[1] - 1)
                        xmeans[state] = xmean

        # Set wdists to None if adding a state was not possible
        if len(unique(wdists)) == 1 and wdists[0, 0] == -1:
            wdists = None

        return wdists

    @staticmethod
    def _wdists_blocks(deltas: ndarray, states: ndarray, x: ndarray, z: ndarray, statewise: bool,
                       blocksize: int) -> ndarray:

        if len(unique(states)) > 1:
            boundopt = zeros(max(states) + 1)
            stateopt = zeros((max(states) + 1, 2))

            prevstate = -1
            for s in unique(states):
                state = where((states > prevstate) & (states <= s))[0]
                numt = state.shape[0]
                if numt > blocksize or s == max(states):
                    xt = x[state]
                    zt = z[state]

                    if statewise:
                        wdists_s = GSBS._wdists_state(deltas=deltas[state], states=states[state], x=xt, z=zt)
                        if wdists_s is None:
                            stateopt[s, :] = [0, 0]
                        else:
                            stateopt[s, :] = unravel_index(wdists_s.argmax(), (wdists_s.shape[0], wdists_s.shape[0])) + \
                                             state[0]

                    wdists = GSBS._wdists(deltas=deltas[state], states=states[state], x=xt, z=zt)

                    boundopt[s] = wdists.argmax() + state[0]
                    prevstate = s

            if statewise and not (stateopt is None):
                stateopt = stateopt[~all(stateopt == 0, axis=1)]
                stateopt = stateopt.astype(int)
            boundopt = boundopt[boundopt > 0]
            boundopt = boundopt.astype(int)

        else:
            boundopt = None
            stateopt = None

        wdists = GSBS._wdists(deltas=deltas, states=states, x=x, z=z, boundopt=boundopt)

        if statewise:
            wdists_s = GSBS._wdists_state(deltas=deltas, states=states, x=x, z=z, stateopt=stateopt)
        else:
            wdists_s = 0

        return wdists, wdists_s

    @staticmethod
    def _zscore(x: ndarray) -> ndarray:
        return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True, ddof=1)
