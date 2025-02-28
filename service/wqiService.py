from pyspark.sql import *
from pyspark.sql.types import *

def calculate_wqi(ph, turbidity, temperature):
    if ph is None or turbidity is None or temperature is None:
        return None
    wqi = (7.0 - abs(ph - 7)) * 25 + (10.0 - turbidity) * 10 + (35.0 - temperature) * 2
    return wqi

