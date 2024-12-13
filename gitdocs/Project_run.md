### Настройка проекта

Необходимые требования к железу:

- Linux OS
- GPU RAM 15+
- RAM 16+
- НЕ ГАРАНТИРУЕМ РАБОТУ В GOOGLe colab

1. Необходимо запустить `make install_all`, который автоматически установит cuda библиотеки
1. Для baseline решение установить библиотеки из файла `requirements/requirements-llama-parser.txt`
1. Для полной работы необходимо 2 API ключа
   1. Huggigface api с доступом в llama-3-8B-Instruct
   1. Для работы LLaMaParse нужен LLAMA_CLOUD_API_KEY
   1. Создаем .env файл и записываем туда API ключи в формате *переменная=ВАШ_КЛЮЧ*
1. Ретривер находится в папке src\\RAG
1. Выбираете какой код из двух запустить
   1. `llama-solo.py` QA RAG bot с использованием преобразования таблицы в HTML
   1. `llama_module.py` QA RAG bot с использованием LLaMaParse(baseline)
1. Для просмотра и расчета метрик запустите `metrics_eval.pynb`
