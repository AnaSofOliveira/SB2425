
## ECG


![[Pasted image 20240930171901.png]]


### End-to-end application
![[Pasted image 20240930171941.png]]
### Baseline
- **Causas:** Movimento dos eletrodos ou movimento da pessoa 

### Remoção dos 50Hz
- Provocado pela corrente elétrica
- Podemos filtrá-lo através de uma média, mas os picos do sinal vai ficar arredondados e, por isso, perdemos informação que pode ser importante. 
- No diagrama o OUTLIER REMOVAL está a verificar se a informação entre 2 picos R consecutivos é importante ou não. 

### QRS DETECTION






### Identificar o pico R
- Calcular a primeira derivada do sinal através do declive
$$ m = \Delta y / \Delta x = (y2 - y1) / (x2 - x1) =  (y2 - y1) / \Delta Ts$$




- **Remover os 50Hz** - Notch filtering - Filtro de rejeição da banda 
	- Método iirnotch do scipy - controlado por um fator de qualidade (quality_factor) que determina de o filtro tem maior ou menor variância 
	- $$ y(n) = b_0 * X(n) +  b_1 * X(n - 1) + ... + b_n * X(n - n) - a_1 * Y(n-1) - a_2 * Y(n-2) - ...  a_M * Y(n-M) -- $$
### Baseline remover
- Tipicamente a baseline é caracterizada por frequências entre 1Hz (high-pass filter) e 40 Hz (low-pass filter)
- Filtro passa alto 
- fiwin function do scipy
 ``` Exemplo:
 numtaps = 3
 f = 0.1
 signal.firwin(numtaps, f)
```

### Deteção de picos 
- Através do Biospy


### Segmentação 
- Da posição de cada pico 200ms para trás e 400ms para a frente

