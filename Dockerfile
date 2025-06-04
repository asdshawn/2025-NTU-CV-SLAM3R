FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

EXPOSE 7860/tcp

RUN apt update && apt -y install python3-pip sudo libglib2.0-dev libgl1 ffmpeg

RUN useradd -rm -d /home/user -s /bin/bash -g root -G sudo -u 1000 user

RUN mkdir -p /home/user/slam3r /home/user/.cache/huggingface
WORKDIR /home/user/slam3r

RUN pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

COPY requirements*txt .

RUN pip3 install xformers==0.0.28.post2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install -r requirements.txt && \
    pip3 install -r requirements_optional.txt

COPY . .

RUN cd slam3r/pos_embed/curope && \
    python3 setup.py build_ext --inplace

RUN echo 'from slam3r.models import Local2WorldModel, Image2PointsModel' > download_models.py && \
    echo 'Image2PointsModel.from_pretrained("siyan824/slam3r_i2p", cache_dir="/home/user/.cache/huggingface")' >> download_models.py && \
    echo 'Local2WorldModel.from_pretrained("siyan824/slam3r_l2w", cache_dir="/home/user/.cache/huggingface")' >> download_models.py && \
    python3 download_models.py

RUN rm download_models.py

# 設定目錄權限並切換使用者
RUN chown -R user:root /home/user/slam3r /home/user/.cache/huggingface
USER user

ENTRYPOINT ["python3", "app.py"]