# Intellegent Placer
## Постановка задачи

Необходимо по фотографии, на которой расположены один или несколько предметов и белый лист с нарисованным многоугольником, определить, помещаются ли предметы в заданный многоугольник. 

### Вход\Выход
- Вход - фотография в формате .jpg, на которой изображён предмет(-ы) на светлом однотонном фоне и белый лист с нарисованным многоугольником.
- Выход - строка в формате [Имя входной фотографии][Ответ]. В случае успеха расположения: Ответ=True, иначе: Ответ=False. Так-же Ответ=Error описывает ситуацию, в которой входное изображение не удовлетворяет требованиям.

## Требования

### Общие требования к фотографиям
- Файл в формате .jpg
- Фотографии без наложения каких-либо фильтров.
- Фотографии сделаны в перпендикулярном направлении к плоскости, угол наклона не больше 10 градусов.
- Фотографии сделаны на одно устройство с одинаковым освещением.
- Объекты имеют чёткую границу, а их тени имеют малую интенсивность.

### Требования к предметам
- Предметы явно выделены относительно поверхности и белого листа бумаги.
- Предметы не могут пересекаться и иметь общих границ(расстояние не меньше 5мм).
- Предметы могут быть представлены в единственном экземпляре.
- Предметы помещаются на лист бумаги формата A4.

### Требования к поверхности
- Поверхность должна быть ровной, однородной и горизонтальной.
- Для всех фотографий используется одна и та же поверхность.

### Требования к исходным данным
- Предметы расположены в центре белого листа бумаги формата А4.
- Края листа отчётливо видны на фотографии.
- Исходные данные содержат 10 фотографий, на каждой по одному различному предмету.

### Требования ко входным данным
- Изображение содержит только заранее известные предметы и многоугольник.
- Многоугольник расположен выше всего набора предметов и не пересекается с ними.
- Многоугольник нарисован на белом листе формата A4 с толщиной менее 5мм.
- Многоугольник является выпуклым и имеет меньше 7 вершин.

## Набор данных

Изображения исходных объектов доступны по [ссылке](https://drive.google.com/drive/folders/1_Az5gIQZRQkBOXIJTfX5e9_Pgxa7kO-a?usp=sharing)

Примеры входных данных расположены в папках "true" , "false" , "error" , в соответствии с ожидаемым результатом работы алгоритма, доступны по следующей [ссылке](https://drive.google.com/drive/folders/11lU9_B2_3u90xtG1A0672gMMpnqoJw9J?usp=sharing).
Так-же в каждой папке присутствует описание для каждой фотографии




 
