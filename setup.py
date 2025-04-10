from setuptools import setup, find_packages
import os
import site
import shutil

# Path to .pth file
pth_file = 'fib_cycles.pth'

# Get the site-packages directory
site_packages = site.getsitepackages()[0]

# Copy the .pth file to site-packages
if os.path.exists(pth_file):
    try:
        shutil.copy(pth_file, os.path.join(site_packages, pth_file))
        print(f"Copied {pth_file} to {site_packages}")
    except Exception as e:
        print(f"Could not copy {pth_file} to {site_packages}: {e}")

setup(
    name="fib_cycles_system",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    description="Fibonacci Harmonic Trading System",
    author="Vijji",
    author_email="vijji@example.com",
    data_files=[
        ('', ['fib_cycles.pth']),
    ],
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "plotly>=4.9.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "streamlit>=1.0.0",
        "yfinance>=0.1.63",
        "talib-binary>=0.4.19",
        "scikit-learn>=0.24.0",
        "python-telegram-bot>=13.0",
        "joblib>=1.0.0",
        "dataclasses-json>=0.5.2",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "jinja2>=3.0.0",
        "python-dotenv>=0.19.0",
        "pytz>=2021.1",
        "PyJWT>=2.1.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0",
        "kaleido>=0.2.1",
        "tqdm>=4.62.0",
        "SQLAlchemy>=1.4.0",
    ],
)