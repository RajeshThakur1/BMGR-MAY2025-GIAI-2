
Docker 

https://docs.docker.com/desktop/setup/install/mac-install/
https://docs.docker.com/desktop/setup/install/windows-install/


Qdrant database 

docker run -p 6333:6333 -p 6334:6334   -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant

—------------


create a saperate enviorenment
-------------------------
conda create -n < name of env> python=3.13 -y

or 

python -m venv <env_name>


or 

uv 

or 

poetry 



To activate the env


mac/linux 

source <env_name>/bin/activate

Window

<env_name>/Scripts/activate




