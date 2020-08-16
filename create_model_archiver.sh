MODEL_STORE="model_store"
if [ ! -d "$MODEL_STORE" ]; then
    mkdir "$MODEL_STORE"
fi

torch-model-archiver \
    --model-name cartoonizer \
    --version 1.0 \
    --model-file ./cartoonizer/net.py \
    --export-path "$MODEL_STORE" \
    --serialized-file ./cartoonizer/weights.pt \
    --handler ./cartoonizer/handler.py \
    --extra-files ./cartoonizer/utils.py,./cartoonizer/guided_filter.py \
    --f \
    2>&1