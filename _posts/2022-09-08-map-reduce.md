---
title: "[Spark] Map Reduce"
layout: post
categories:
- Spark
tags:
- Spark
---

## Big Data
빠르게 증가하고 구조화/구조화 되어 있지 않은 현재 DB 툴로 처리하기 곤란한 데이터    
크기, 계산 복잡도 : 빅데이터 정의하는 척도  

**3V**
- Volume(크기)
- Variety(다양성)
- Velocity(빠르게 생성되고 유입)

**빅데이터 분석**    
급격히 증가하는 대용량의 데이터, 다양한 데이터 타입에서 숨겨진 패턴, 상관관계, 유용한 정보 뽑아냄

Big Data - 데이터만 커진 것 아닌가?
- 데이터 커지는 것이 질적으로 어떤 변화 가져오는지, 어떤 어려움 있는지

### Bonferroni's Principle
테러리스트가 있고 1000일간에 서로 다른 두 날에 두 사람이 같은 호텔 숙박한다고 함

**가정**
- 10억명이 테러리스트 될 수도 있다.
- 모든 이가 100일에 하루는 호텔에 간다.
- 호텔 수용 인원은 100명이고 10만개의 호텔이 있다.
- 1000일을 조사한다.

- A와 B가 같은 두 날에 같은 호텔에 있을 확률 : $({1 \over 100} \times {1 \over 100} \times {1 \over 10^5})^2={1 \over 10^{18}}$
- 사람의 쌍 : ${10^9 \choose 2}= 5 \cdot 10^{17}$
- 날의 쌍 : ${10^3 \choose 2} = 5 \cdot 10^5$
위의 수를 모두 곱하면 25만 쌍의 결과가 나온다.

### Age of Parallelism
슈퍼 컴퓨터 : CPU 열, 메모리 많이 달아도 속도의 한계 - 여러 개 컴퓨터를 연결해서 쓰자(병렬 컴퓨터)

### Programming Model
Functional programming
- map 함수 (key, value) -> (key, value) 복수개
- reduce 함수 (key, value list) -> (key, value) 출력

### Map/Reduce Example #1(word counting)
```
slave 1    apple cat banana (ant1)
slave 2    cat banana (ant2)
                              apple banana (ant3)
slave 3    apple banana (ant4)
```
- 각 개미는 라인 하나씩 읽음
- hash table 사용 불가

map 함수에 넣어야 함
```
tokens <- tokenize(line)
foreach (word in tokens)
{
    emit(word, 1); # word : key, 1 : value 출력
}
```

개미(map 함수) - 병렬적으로


**slave 1** 
apple cat banana 
- (ant1)  <apple, 1> <cat, 1> <banana, 1>

**slave 2**  
cat banana 
apple banana 
- (ant2)  <cat, 1> <banana, 1>
- (ant3)  <apple, 1> <banana, 1>

**slave 3** 
apple banana 
- (ant4)  <apple, 1> <banana, 1>


**slave 1**  
<apple, 1>  
<apple, 1>      (dung beetle 1) -> <apple, 3>  
<apple, 1>  

**slave 2**    
<banana, 1>  
<banana, 1>  
<banana, 1>     (dung beetle 2) -> <banana, 4>  
<banana, 1>  

**slave 3**    
<cat, 1>  
<cat, 1>        (dung beetle 3) -> <cat, 2>
															
shuffling                   Reduce
															


### Resilent distributed datasets(RDD)
- 변형 불가능(값 고칠 수 없음 -> 바꾸려면 새 RDD로 transformation 해줘야 함)
- partitioned (slave로 쪼개져 있어서)
- spark에서 한번에 처리하는 data flow에서 처리하는 데이터 덩어리


------------------
## RDD
### Spark Programming
- 모든 데이터는 RDD로 표현됨.
1. Disk에서 data 가져와 RDD 생성
2. Transformation으로 값의 재구성
3. RDD에 Action(Aggregation)으로 값을 화면에 출력하거나 Disk에 저장

### SparkContext vs. SparkSession
**Spark Context**
- Spark cluster에 접근하기 위한 통로
- 각종 설정 세팅, RDD 작성 및 조작되는 API 제공
```python
import pyspark
sc = pyspark.SparkContext()
```

**Spark Session**
cluster Manager 등에 접근하기 위한 통로 역할. 설정, RDD 제공하는 API 패키지
- 다양한 Context (SQLContext, HiveContext(), etc) -> 표준 API 필요
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('appname').getOrCreate()
```

### Spark의 병렬처리
- 디스크에 있는 파일을 메모리에 로드
```python
data_from_file = sc.textFile('hdfs://~.txt')
# RDD               RDD 생성   DB, MySQL -> RDD
```
메모리에 분리되어 로드되고 하나의 RDD로 통합처리

**transform** ( filter ) RDD 새로 생김
```
data_filtered = data_from_file.filter(lambda row : row[16] == "2014")
```

### RDD
- RAM에 저장된 각 블록을 Partition이라고 하고 RDD는 Partition들의 집합

### Example Job
```python
sc = SparkContext("local", "MyJob")
file = sc.textFile("/...") # RDD
errors = file.filter(lambda x : "ERROR" in x) # RDD | filter : transformation
errors.count() # Action
```

### Spark Operations
- Transformations(define a new RDD) : map, filter, sample, groupByKey, reduceByKey, sortByKey, flatMap, union, join, cogroup, cross, mapValues
- Actions : collect, reduce, count, save, lookupKey
