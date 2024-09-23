Pruebas para TensorRT-LLM

* Instalación

Es IMPORTANTE Crear entorno virtual con python 3.10 e instalar CUDA 12.1-12.4 de lo contrario no servirá.

Usar requirements.txt o lo siguiente para dependencias (hay varias maneras pero ninguna me funcionó, esta combinación funcionó correctamente):

** Instalar este par de bibliótecas:

pip install mpi4py
pip install impi

** Instalar torch: 
pip install torch==2.4.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

** Si se demora mucho tiempo o/y consume bastante memoria RAM puede usarse: 
pip install torch==2.4.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

** Luego instalar biblioteca:
pip install tensorrt_llm==0.12.0 --extra-index-url https://pypi.nvidia.com --no-cache-dir


	 
