def print_model_info(): 
    info = """
GaussianModel()
# Dynamics
# --------
    f       : Callable  # forward function (must be numpy compatible) 
    g       : Callable  # observation function (must be numpy compatible) 

    fsymb   : Callable  # symbolic declaration of f using sympy
    gsymb   : Callable  # symbolic declaration of g using sympy

    df      : cdotdict
    d2f     : cdotdict
    dg      : cdotdict
    d2g     : cdotdict

# Model dimensions
# ----------------
    m  : int    # number of inputs
    n  : int    # number of states
    l  : int    # number of outputs
    p  : int    # number of parameters

# Parameters 
# ----------
    pE : np.ndarray     # prior expectation of parameters p
    pC : np.ndarray     # prior covariance of parameters p

# Hyperparameters
# ---------------
    hE : np.ndarray         # prior expectation of hyperparameters h (log-precision of cause noise)
    hC : np.ndarray         # prior covariance of hyperparameters h (log-precision of cause noise)

    gE : np.ndarray         # prior expectation of hyperparameters g (log-precision of state noise)
    gC : np.ndarray         # prior covariance of hyperparameters g (log-precision of state noise)

    Q  : List[np.ndarray]   # precision components (input noise)
    R  : List[np.ndarray]   # precision components (state noise)

# Model specification
# -------------------
    V  : np.ndarray     # fixed precision (input noise)
    W  : np.ndarray     # fixed precision (state noise)

    sv : np.ndarray     # smoothness (input noise)
    sw : np.ndarray     # smoothness (state noise)

    x  : np.ndarray     # explicitly specified states
    v  : np.ndarray     # explicitly specified inputs

    xP : np.ndarray     # precision (states)
    vP : np.ndarray     # precision (inputs)

    constraints: np.ndarray

    # if not none, contains the function to compute the delay matrix or 
    # the delay matrix itself, which must broadcast to the system jacobian
    delays 
    delays_idxs
    """
    print(info)