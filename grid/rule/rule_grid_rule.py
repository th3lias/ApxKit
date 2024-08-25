#  Created 2024. (Elias Mindlberger)

#  Created 2024. (Elias Mindlberger)
from grid.rule.rule import GridRule


class RuleGridRule(GridRule):
    CLENSHAW_CURTIS = 'clenshaw-curtis'
    CLENSHAW_CURTIS_ZERO = 'clenshaw-curtis-zero'
    FEJER2 = 'fejer2'
    RLEJA = 'rleja'
    RLEJA_ODD = 'rleja-odd'
    RLEJA_DOUBLE2 = 'rleja-double2'
    RLEJA_DOUBLE4 = 'rleja-double4'
    RLEJA_SHIFTED = 'rleja-shifted'
    RLEJA_SHIFTED_EVEN = 'rleja-shifted-even'
    MAX_LEBESGUE = 'max-lebesgue'
    MAX_LEBESGUE_ODD = 'max-lebesgue-odd'
    MIN_LEBESGUE = 'min-lebesgue'
    MIN_LEBESGUE_ODD = 'min-lebesgue-odd'
    LEJA = 'leja'
    LEJA_ODD = 'leja-odd'
    MIN_DELTA = 'min-delta'
    MIN_DELTA_ODD = 'min-delta-odd'
