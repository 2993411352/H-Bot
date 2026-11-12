FROM horizonrobotics/robo_orchard_sim@sha256:69e64ff85395140eb1f5d058c906be6a0d664a1b9771b047db8641651e699710
USER root
ARG CUDA_ARCH
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates xvfb libegl1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY . /opt/hbot
RUN git clone https://github.com/HorizonRobotics/RoboOrchardSim.git /opt/hbot/sim && \
    git -C /opt/hbot/sim checkout --detach f2e041e5fabe354c1398ea97c934f603bb65d14d && \
    git -C /opt/hbot/sim apply --check /opt/hbot/patches/collector.patch && \
    git -C /opt/hbot/sim apply /opt/hbot/patches/collector.patch && \
    python3 -m venv --system-site-packages /opt/hbot/venv && \
    /opt/hbot/venv/bin/python -m pip install --no-deps --no-build-isolation /opt/hbot/sim && \
    /opt/hbot/venv/bin/python -m pip install --no-deps -r /opt/hbot/converter-requirements.txt && \
    /opt/hbot/venv/bin/python /opt/hbot/build_curobo.py "$CUDA_ARCH"
ENV PATH=/opt/hbot/venv/bin:$PATH \
    PYTHONPATH=/opt/hbot/runtime:/opt/hbot/sim:/opt/hbot/curobo/src \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ORCHARD_ASSET=/assets ORCHARD_ASSET_LIBRARY=/assets/OBJECTS_v2 \
    ASSETS_DIR=/assets/OBJECTS_v2 NV_ASSET_ROOT_DIR=/assets/NVIDIA/Assets/Isaac/4.1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    VK_ICD_FILENAMES=/opt/hbot/vulkan.json \
    __EGL_VENDOR_LIBRARY_FILENAMES=/opt/hbot/egl.json
WORKDIR /opt/hbot/sim
RUN /opt/hbot/venv/bin/python /opt/hbot/runtime/cpu_check.py
ENTRYPOINT ["/opt/hbot/venv/bin/python"]
CMD ["/opt/hbot/runtime/inventory.py"]
