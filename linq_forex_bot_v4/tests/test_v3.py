from src.optimizer import reward_grid, evaluate_ratio
import pandas as pd

def test_grid():
    assert reward_grid(1,2,0.25)==[1.0,1.25,1.5,1.75,2.0]

def test_ratio_evaluation():
    setups=pd.DataFrame({"max_favorable_r":[3.2,0.5],"final_r":[-1.0,-1.0]})
    r=evaluate_ratio(setups,3.0)
    assert r["net_r"]==2.0
