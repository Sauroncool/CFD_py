import numpy as np

# Initial Condition Functions
def init_cond_1(x):
    return 1.0 if x <= 0.2 else 0.0

def init_cond_2(x):
    if x < 0.05:
        return 0.0
    elif 0.05 <= x < 0.35:
        return np.sin(4 * np.pi * (x - 0.05) / 0.3)
    else:
        return 0.0

def init_cond_3(x):
    if x < 0.05:
        return 0.0
    elif 0.05 <= x < 0.35:
        return np.sin(8 * np.pi * (x - 0.05) / 0.3)
    else:
        return 0.0

def init_cond_4(x):
    if x < 0.05:
        return 0.0
    elif 0.05 <= x < 0.35:
        return np.sin(12 * np.pi * (x - 0.05) / 0.3)
    else:
        return 0.0

def init_cond_5(x):
    if x < 0.05:
        return 0.0
    elif 0.05 <= x < 0.35:
        return np.sin(4 * np.pi * (x - 0.05) / 0.3)
    else:
        return 0.0

# Boundary Condition Functions
def boundary_cond_1():
    return {"left": 1.0, "right": 0.0}

def boundary_cond_2():
    return {"left": 0.0, "right": 0.0}

# Test Case Functions
def test_case_1(x_values):
    return {
        "initial_condition": [init_cond_1(x) for x in x_values],
        "boundary_conditions": boundary_cond_1(),
    }

def test_case_2(x_values):
    return {
        "initial_condition": [init_cond_2(x) for x in x_values],
        "boundary_conditions": boundary_cond_2(),
    }

def test_case_3(x_values):
    return {
        "initial_condition": [init_cond_3(x) for x in x_values],
        "boundary_conditions": boundary_cond_2(),
    }

def test_case_4(x_values):
    return {
        "initial_condition": [init_cond_4(x) for x in x_values],
        "boundary_conditions": boundary_cond_2(),
    }

def test_case_5(x_values):
    return {
        "initial_condition": [init_cond_5(x) for x in x_values],
        "boundary_conditions": boundary_cond_2(),
    }

def test_cases(x_values):
    return [test_case_1(x_values), test_case_2(x_values), test_case_3(x_values), test_case_4(x_values), test_case_5(x_values)]
