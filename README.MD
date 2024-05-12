Customer Cluster and Fraud Propensity API
Descrição
Este serviço, construído com Flask, é projetado para segmentar clientes em clusters baseados em seus dados financeiros e comportamentais, além de avaliar a propensão à fraude. Os clusters são etiquetados com personas que descrevem características predominantes. O serviço é empacotado em um contêiner Docker, garantindo facilidade na distribuição e consistência entre os ambientes de desenvolvimento e produção.

Pré-requisitos
Antes de começar, certifique-se de ter o Docker instalado em sua máquina. Caso não possua, instale-o seguindo as instruções disponíveis no site oficial do Docker.

Instalação
Para configurar a aplicação em seu ambiente local, siga os passos abaixo.

Clonar o Repositório
Primeiramente, clone o repositório onde o código da aplicação está disponível:

```bash
git clone https://github.com/fspoliveira/ClusterAndClassification
cd ClusterAndClassification
```

## Como Usar

Siga as instruções abaixo para colocar a aplicação Model Manager em funcionamento usando Docker.

### Construção da Imagem Docker

Abra o terminal e navegue até o diretório onde está o Dockerfile. Execute o comando abaixo para construir a imagem Docker do projeto:

```bash
docker build -t cd customer-cluster .
```

Depois execute

```bash
docker run -d -p 5002:5002 --network bridge cd customer-cluster
```