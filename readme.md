
387
```python
388
y = data['TARGET CLASS']
389
```
390
​
391
​
392
```python
393
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
394
```
395
​
396
## KNN model deployment.
397
​
398
​
399
```python
400
from sklearn.neighbors import KNeighborsClassifier
401
```
402
​
403
​
404
```python
405
knn = KNeighborsClassifier(n_neighbors=1)
406
```
407
​
408
​
409
```python
410
knn.fit(X_train,y_train)
411
```
412
​
413
​
414
​
415
​
416
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
417
               metric_params=None, n_jobs=None, n_neighbors=1, p=2,
418
               weights='uniform')
419
​
420
​
421
​
422
​
423
```python
424
predictions = knn.predict(X_test)
425
```
426
​
427
## Model Evaluation
428
​
429
​
430
```python
431
from sklearn.metrics import classification_report,confusion_matrix
432
```
433
​
434
​
435
```python
436
print(confusion_matrix(y_test,predictions))
437
print("___"*20)
438
print(classification_report(y_test,predictions))
439
```
440
​
441
    [[134   8]
442
     [ 11 147]]
443
    ____________________________________________________________
444
                  precision    recall  f1-score   support
445
    
446
               0       0.92      0.94      0.93       142
447
               1       0.95      0.93      0.94       158
448
    
449
       micro avg       0.94      0.94      0.94       300
450
       macro avg       0.94      0.94      0.94       300
451
    weighted avg       0.94      0.94      0.94       300
452
    
453
​
454
​
