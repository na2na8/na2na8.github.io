---
title: "[Spark] Action / Transformation"
layout: post
categories:
- Spark
tags:
- Spark
---

parallelize 결과 정확하지 않음. 파티션에 따라 어떻게 나오는지 하려고 임의로 둔 것.
## Action
### reduce(func)
- 함수로 RDD를 단일 객체가 될 때까지 줄임
``` python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
data_reduce = data.reduce(lambda x, y : x + y) # [6] [15] [34] -> 55
print(data_reduce) # 55
```

### .collect()
- 데이터 셋 전체를 리스트로 반환
```python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
data.collect() #[1,2,3,4,5,6,7,8,9,10]
```

### .count()
- RDD element 개수
```python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
data.count() # [3] [3] [4] -> 10
```

### .first()
- 처음 1개의 element
```python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
data.first() # 1 ... 1번 파티션의 첫 번째 element driver로 가져옴
```

### .take(n)
- 처음 n개의 elements를 리스트로 반환
```python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
data.take(7) # [1,2,3,4,5,6,7]
```

### .takeSample()
- RDD에서 랜덤한 샘플 뽑아 드라이버로 반환
- takeSample(withReplacement, num, seed=None)
	- withReplacement : 복원(True), 비복원(False) 뽑기 선택
	- num : sample 크기
	- seed : random seed 값
```python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
data.takeSample(False,3) # [1, 4, 9]
```

### .takeOrdered()
- RDD에서 작은 순서대로 지정한 개수 리턴. 매우 큰 데이터에서는 sort() 사용하기
```python
data = sc.parallelize([10,9,8,7,6,5,4,3,2,1], 3) # [10, 9, 8] [7, 6, 5, 4] [3, 2, 1]
data.takeOredered(3) # [1, 2, 3]
```

### .saveAsTextFile(path)
- 데이터 셋 파티션 하나당 하나의 파일로 저장
	- path : RDD가 저장될 위치
	- local filesystem, HDFS 등
```python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
data.saveAsTextFile("data")
```

### .countByKey()
- key-value 쌍으로 이루어진 RDD에서 작동. key 기준으로 개수 셈
- 각 key의 count를 갖는 dictionary 리턴
```python
data = sc.paralleize([('a',1),('b',2),('c',3),('d',4)])
data.countByKey() # defaultdict(int, {'a':2, 'b':1, 'c':1})
```

## Transformation
### Transformation
- RDD -> RDD로 변경
- ex) map(), flatmap(), sample(), reduceByKey()

### map(func)
- RDD 각 element에 func을 적용하여 새로운 RDD 리턴
```python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
data = data.map(lambda x : x + 1)
data.collect() # [2,3,4,5,6,7,8,9,10,11]
```

### filter(func)
- .filter()는 조건 통과한 값만 출력
```python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
filtered = data.filter(lambda x : x < 5)
filtered.collect() # [1, 2, 3, 4]
```

### flatMap(func)
-  map()과 비슷. element 하나 받아 0, 1, 또는 복수개 출력 가능
-  map() 하고 flatten
```python
data = sc.parallelize([1,2,3,4], 1)
data.map(lambda x : [x, x*x]).collect() # [[1,1],[2,4],[3,9],[4,16]]
data.flatMap(lambda x : [x, x*x]).collect() # [1, 1, 2, 4, 3, 9, 4, 16]
```

### mapPartitions(func)
- RDD 각 파티션에 대해 map tlfgod
```python
data = sc.parallelize([1,2,3,4,5,6,7,8,9,10], 3) # [1, 2, 3] [4, 5, 6] [7, 8, 9, 10]
```
```python
def f(x) :
	yield sum(x)
```
```python
data.mapPartitions(f).collect() # [6, 15, 34]
```

### mapValues(func)
- key-value 쌍 RDD에 대해 값에 대해서만 func 적용
```python
x = sc.paralleize([("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])])
def f(x) : return len(x)
x.mapValues(f).collect() # [("a", 3), ("b", 1)]
```

### reduceByKey(func, numPartitions)
- key별로 func tngod
- numPartitions(Optional) : 결과로 생성되는 RDD 파티션 개수
```python
data = sc.parallelize([('a',1),('b',2),('c',3),('a',4)])
print(data.reduceByKey(lambda x, y : x + y).collect()) #[('b',2), ('c',3),('a',5)]
```

## Word Count
```python
text = "Spark is fast"
data = text.flatMap(lambda line : (line.split(' ')) # ['spark', 'is', 'fast']
```
```python
data = data.map(lambda word : (word, 1)) # [('spark',1), ('is',1), ('fast',1)]
```

추가로 text에 "spark is easy" 라는 문장이 있어 이에 대해서도 위의 함수 실행했다고 한다면
```python
[('spark',1), ('is',1), ('fast',1)] [('spark',1), ('is',1), ('easy',1)]
```

```python
data = data.reduceByKey(lambda a, b : a+b) # [('spark',2), ('is',2), ('fast',1), ('easy',1)]
```
