#!/usr/bin/env fish
# -------------------------------------------------
# Cmpi‑y – Customer Churn Prediction project setup
# -------------------------------------------------
# Usage:
#   ./setup.fish            # interactive setup
#   ./setup.fish notebook   # run the Jupyter notebook
#   ./setup.fish app        # launch the Streamlit app
# -------------------------------------------------

# ---- 1️⃣ Create / activate virtual environment ----
function _make_venv
    echo "Creating virtual environment …"
    python -m venv venv --system-site-packages
    source venv/bin/activate.fish
end

# If the venv does not exist, create it
if not test -d venv
    _make_venv
else
    # activate existing venv
    source venv/bin/activate.fish
end

# ---- 2️⃣ Upgrade pip and install dependencies ----
echo "Upgrading pip and installing Python packages …"
pip install --upgrade pip
pip install \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyterlab \
    streamlit \
    joblib

# ---- 3️⃣ Helper functions -------------------------------------------------
function _train
    echo "Training the model …"
    python train_model.py
end

function _run_notebook
    _train
    echo "Launching Jupyter Lab …"
    jupyter lab notebook/Customer_Churn_Prediction.ipynb
end

function _run_app
    _train
    echo "Starting Streamlit app …"
    streamlit run app/app.py
end

# ---- 4️⃣ Command‑line interface -------------------------------------------
switch $argv[1]
    case notebook
        _run_notebook
    case app
        _run_app
    case ''
        echo "Setup complete. Use one of these commands next:"
        echo "  ./setup.fish notebook   # run the notebook"
        echo "  ./setup.fish app        # launch the Streamlit UI"
    case '*'
        echo "Unknown argument: $argv[1]"
        echo "Valid options are: notebook, app"
end
