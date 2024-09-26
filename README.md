Pruebas para TensorRT-LLM

# Pasos para instalar dependencias, repositorio y biblioteca

Es IMPORTANTE Crear entorno virtual con python 3.10 e instalar CUDA 12.1-12.4 de lo contrario no servirá.
Considera tener al menos 150 a 200 GB libres en disco.

* Instalar make y ponerlo como variable de entorno.
* Clonar repositorio: https://github.com/NVIDIA/TensorRT-LLM
* Crear un entorno virtual dentro del repositorio e incializarlo para instalar dependencias.
* pip install mpi4py
* pip install impi
* git lfs install
* git lfs pull
* pip install openmpi-bin
* python -m pip install --upgrade pip
* $env:CUDACXX="ruta_cuda", por ejemplo: $env:CUDACXX="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"                       
* Primera forma de instalar: pip install --extra-index-url https://pypi.nvidia.com tensorrt-llm            
* Segunda forma de instalar: pip install tensorrt_llm==0.12.0
* Tercera forma: Si falla usar versión 0.10.0.

Adicional si algo sale mal, se puede instalar lo siguientem aunque tensorrt_llm ya debería instalar las dependencias.

* pip install --upgrade transformers 
* Si se tiene un problema con torch (se descarga solo al instalar tensorrt_llm), puedes instalar torch desde: pip install torch==2.4.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 (si se demora y consume mucha memoria RAM se puede usar al final --no-cache-dir).
* Instalar los requerimientos desde requirements.txt dentro de la carpeta del repositorio (usando pip install -r requirements.txt)
* Si se va a compilar algún modelo, instalar los requerimientos dentro de la carpeta examples del modelo, por ejemplo, examples/llama, e instalar requirements.txt (también con pip).

# Crear el modelo

* Descargar el modelo original para compilar desde Hugginface. Ejemplo: Meta-Llama-3.1-8B-Instruct
* Si el modelo tiene la carpeta "original" copiar esos archivos a la raíz de la carpeta del modelo.
* Usar el fichero convert_checkpoint con parámetro de entrada hacia el directorio del modelo y salida personalizada, se recomienda usar ckpt al final (checkpoint), ejemplo: 
python convert_checkpoint.py --model_dir C:\Users\Sixpounder\Documents\modelos_autoresumen\llama3inst\Meta-Llama-3.1-8B-Instruct --output_dir llama-3.1-8b-ckpt
* Compilar motor, indicando la ruta de los ficheros que se crearon anteriormente, por ejemplo: trtllm-build --checkpoint_dir llama-3.1-8b-ckpt --gemm_plugin float16 --output_dir ./llama-3.1-8b-engine
* Ahora correr el fichero run apuntando al modelo


# nota
Tuve problemas para instalar dependencias entre la biblioteca y el repositorio oficial, asi que lo hice por separado, una vez que funcionó la biblioteca con el código de prueba en Python, copie los archivos de tensorrt_ll y todos los relacionados a las dependencias del repositorio tensorrt, una vez realizado, funciona correctamente.

# Generar modelo con tensor RT

Para generar modelo con precisión de float16

Los archivos de la carpeta con nombre "original" del modelo llama 3.1, pasarlos o copiarlos a la raíz.

* python convert_checkpoint.py --meta_ckpt_dir [directorio_del_modelo_original] --output_dir [directorio_del_modelo_original]/trt_ckpts/tp8-pp2/ --dtype bfloat16 --tp_size 8 --pp_size 2 --load_by_shard --workers 2

python convert_checkpoint.py --meta_ckpt_dir C:\Users\Sixpounder\Documents\modelos_autoresumen\llama3inst\Meta-Llama-3.1-8B-Instruct --output_dir C:\Users\Sixpounder\Documents\modelos_autoresumen\llama3inst\Meta-Llama-3.1-8B-Instruct/trt_ckpts/tp8-pp2/ --dtype bfloat16 --tp_size 8 --pp_size 2 --load_by_shard --workers 2

* trtllm-build --checkpoint_dir llama-3.1-8b-ckpt --gemm_plugin float16 --max_batch_size 8 --output_dir ./llama-3.1-8b-engine

* python ../run.py --engine_dir [directorio_motor_generado]  --max_output_len 100 --tokenizer_dir [directorio_modelo_original] --input_text "Cómo se dice perro en inglés?"

Ejemplo:

python ../run.py --engine_dir C:\Users\Sixpounder\Documents\TensorRT-LLM\examples\llama\llama-3.1-8b-engine  --max_output_len 100 --tokenizer_dir C:\Users\Sixpounder\Documents\modelos_autoresumen\llama3inst\Meta-Llama-3.1-8B-Instruct --input_text "Cómo se dice perro en inglés?"

	 
# Conclusión

Fue necesario instalar la biblioteca aparte y luego copiar todo lo relacionado a tensorrt_llm para mandarlo al entorno virtual del repositorio completo de tensorrt_llm.

Cuando se intenta hacer la petición al modelo aparece el siguiente error:

ImportError: cannot import name 'supports_inflight_batching' from 'tensorrt_llm._utils' (..\TensorRT-LLM\venv\lib\site-packages\tensorrt_llm\_utils.py)

Se resuelve si vamos a utils.py (directo del directorio examples/llama) y quitamos la línea: from tensorrt_llm._utils import supports_inflight_batching  # noqa

Luego vamos a run.py y quitamos las líneas que tengan: supports_inflight_batching

Si al generar la petición al modelo aparece un desbordamiento de memoria de la GPU, usar batch 1 a 8 al compilarlo en el segundo paso

Con varias pruebas se ve que si es más rápido que llamacpp pero consume más VRAM a pesar de ir moviendo parametros en las compilaciones