# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
# See https://github.com/ucsd-ets/datahub-docker-stack/wiki/Stable-Tag 
# for a list of the most current containers we maintain
FROM $BASE_CONTAINER

# 3) install packages using notebook user
USER jovyan

# RUN conda install -y scikit-learn
RUN pip install --upgrade pip wheel && pip cache purge
RUN pip install --no-cache-dir numpy pandas matplotlib tqdm 

# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]