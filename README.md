# Проектный Хакатон MIFIML команды Space Plans
![My Image](space_plans.jpg)
## Вводные данные
**Название команды:** Space plans

**Структура команды:**

* **Team leader:** Кловер Натан ( Clover9301@gmail.com )

* **Product manager:** Воронин Михаил Михайлович ( mixan2912@gmail.com )

* **Analytics team:** 
  * Прокофьев Михаил Всеволодович ( k0rsair@mail.ru )
  * Жадаев Василий Васильевич ( st1m_97@mail.ru )
  * Колотий Марианна Павловна ( mildburg@rambler.ru )
## Тема проекта - Задача №2 — тематическая классификация текстов
* **Набор данных:** обработанные нами данные voll_data.csv
### Бизнес-постановка задачи
* **Проблема**
  * Современные цифровые каналы ежедневно генерируют большое количество коротких пользовательских сообщений — от комментариев и постов до отзывов и заявок.
  * Эти тексты могут затрагивать разные области: спорт, личную жизнь, политику, рекламу, юмор и социальные сети.

  * В ручном режиме быстро обрабатывать и классифицировать такие сообщения невозможно. Отсутствие автоматизированной системы приводит к:
    * потере времени на разбор и сортировку контента;

    * снижению эффективности реагирования на чувствительные или приоритетные темы (например, политика или негативная реклама);

    * невозможности масштабного анализа пользовательской активности и интересов.

* **Цель**
  * Создать систему, которая определяет вероятность принадлежности текста к одной или нескольким тематикам из заданного списка.
  * Это задача классификации с пересекающимися классами. Один текст может:
    * относиться к нескольким тематикам;
    * не относиться ни к одной из заданных тем.
    * Также необходимо реализовать примитивный веб-интерфейс (UI) для ручного тестирования.


* **Этапы реализации**
  1. **Подготовка и разметка данных**
    * Приведение данных к единому формату
    * Обработка текстов ( удаление лишних запятых, спец.символов, стоп-слов и т.д)
    * Преобразование меток в формат MultiLabelBinarizer
    * Разделение на тренировочную, валидационную и тестовую выборку

  2. **Выбор модели и генерация эмбеддингов**
      * RuBERT
      * TF-IDF
      * FRIDA

  3. **Обучение классификатора**
    * Выбор модели классификатора:
      * Logistic Regression
      * catboost
      * DeepPavlov/rubert-base-cased
    * Подбор гиперпараметров
    * Сохранение выбранной модели и всех преобразователей(tokenizer)

  4. **Оценка качества**
    * Вычисление метрик micro/macro Precision, Recall, F1
    * Построение confusion matrix для каждой темы
    * Анализ ошибок (ложноположительные/ложноотрицательные)

  5. **Разработка веб-интерфейса**
    * Форма для ввода текста и кнопкой для предсказания темы
    * Возможность посмотреть результат в виде класса ( и возможно вероятности )

  6. **Тестирование**
    * Внедрение на этапе разработки тестов assert
    * Проверки на ошибки в работе UI и модели

  7. **Документация и сопровождение**
    * Подготовка README
    * Название, цели и задачи проекта
    * Состав и номер (название) команды
    * Описать этапы и ход работы
    * Раздел про полученные результаты (табличка с метриками(confusion matrix)) и краткие выводы
    * Выделить пояснения в репозитория, что и где находится
    * Приложить инструкцию как её запускать через Докер
    * Приложить интересные графики из EDA и скрины работы вашего приложения

## Установка
В дирректории docker необходимо выполнить следующие команды:
```python
docker compose build
docker compose up -d
```
После чего приложение будет доступно по localhost в браузере.
