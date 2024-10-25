from setuptools import setup

setup(
    name="pepsi",
    packages=[
        "point_e",
        "point_e.diffusion",
        "point_e.evals",
        "point_e.models",
        "point_e.util",
        "pepsi",
    ],
    install_requires=[
        "filelock",
        "Pillow",
        "torch",
        "fire",
        "humanize",
        "requests",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "scipy",
        "numpy",
        "clip @ git+https://github.com/openai/CLIP.git",
        "wandb",
        "pandas",
        "pytorch_lightning",
        "mitsuba",
        "ailia",
        "tensorboard",
    ],
    author="OpenAI",
)
