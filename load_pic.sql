-- 色盲點圖題目存進資料庫的table格式
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    image_data BYTEA
);
