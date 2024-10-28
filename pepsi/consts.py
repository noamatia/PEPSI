ID = "id"
LR = "lr"
DEV = "dev"
PCS = "pcs"
LOSS = "loss"
RUNS = "runs"
CKPT = ".ckpt"
LOCAL = "local"
PARTS = "parts"
TEXTS = "texts"
WANDB = "wandb"
OUTPUT = "output"
SOURCE = "source"
TARGET = "target"
IMAGES = "images"
SPLITS = "splits"
EPOCHS = "epochs"
INDICES = "indices"
N_PTS_ENCODE = 1024
N_PTS_SAMPLE = 4096
PROMPTS = "prompts"
VAL_FREQ = "val_freq"
BASE_DIR = "base_dir"
GUIDANCE = "guidance"
UPSAMPLE = "upsample"
LOG_WANDB = "log_wandb"
COPY_PROB = "copy_prob"
DEBUG_LOG = "debug.log"
WANDB_DIR = "WANDB_DIR"
INJECTIONS = "injections"
BATCH_SIZE = "batch_size"
COPY_PROMPT = "copy_prompt"
CHECKPOINTS = "checkpoints"
SHAPENET_DIR = "shapenet_dir"
MODEL_NAME = "base40M-textvec"
WANDB_PROJECT = "wandb_project"
INIT_VAL_DATA = "init_val_data"
VAL_DATALOADER = "val_dataloader"
COND_DROP_PROB = "cond_drop_prob"
TARGET_LATENTS = "target_latents"
SOURCE_LATENTS = "source_latents"
POINTNET_MODELS = "pointnet_models"
CHECKPOINT_PATH = "checkpoint_path"
UTTERANCE_KEY_CONF_KEY = "utterance_key"
SHAPETALK_CSV_PATH = "data/shapetalk.csv"
SHAPE_CATEGORY_CONF_KEY = "shape_category"
PREV_CHECKPOINT_PATH = "prev_checkpoint_path"
PARTNET_REMOTE_DIR = "https://storage.googleapis.com/ailia-models/pointnet_pytorch"


# https://github.com/hasancaslan/BeautifulPointCloud
class XMLTemplates:
    HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1000"/>
            <integer name="height" value="1000"/>
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""
    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""
    TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""
