# wine_quality
Estrutura do código :

Foram definidas, no início, os pacotes importados, configurações para elaboração do código via Jupyter Notebook, a função para verificação da acurácia e variância dos modelos. Depois realizou-se a importação dos dados, verificaram-se erros na variável "alcohol", e depois de consertar tais erros realizou-se a análise exploratória dos dados. Observando-se as variáveis principais pelo correlograma, foram então aplicados os métodos de cada modelo (no total 5 : ANN,GNB,MLR,RFC,DTC). Por fim, verificou-se o melhor modelo dentre os analisados.

Respostas às perguntas:

a) Como foi a definição da sua estratégia de modelagem?

Resposta : O primeiro passo foi identificar os inputs e outputs e determinar se era um problema de classifcação ou de regressão. Uma vez detectado que o problema era de classificação, a estratégia foi aplicar os modelos padrões de classificação no contexto de machine learning e deep learning.

b) Como foi definida a função de custo utilizada?

Resposta : A função de custo utilizada foi a de perda pelo erro quadrático médio (Mean Square Error), a fim de analisar o desempenho dos modelos.

c) Qual foi o critério utilizado na seleção do modelo final?  

Resposta : Foi justamente a combinação de acurácia associada ao erro médio quadrático do modelo. Modelos com grande acurácia e elavado MSE foram excluídos (pela baixa confiabilidade) e tiveram preferência modelos com razoável acurácia e baixo MSE. 

d) Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar
este método?

Resposta: Para a validação dos modelos, foram estabelecidas rotinas de validação via métodos de grid search e cross-validation. Estes métodos serviram para indicar a presença de overfitting dos modelos e através deles identificar os modelos com maiores variações.

e) Quais evidências você possui de que seu modelo é suficientemente bom?

Resposta: Após ter sido realizado o correlograma foi verificado a presença de uma variável independente com alto nível de correlação com as demais e portanto esta foi excluída do modelo escolhido que no caso foi o Artificial Neural Networks. A aplicação de Principal Component Analysis foi descartada pelo fato de que apenas uma variável apresentava tal comportamento, em termos de correlação. Após a seleção das variáveis do modelo e de tê-lo rodado, verificou-se uma razoável acurácia e baixíssimo MSE. Logo, o modelo pode ser considerado suficientemente bom.
   
