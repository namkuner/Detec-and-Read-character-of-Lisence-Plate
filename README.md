# Nhan dien va read bien so xe may (Detect License Plate)
Các bước thực hiện
1. Kiếm data: data này mình săn lùng khắp nơi! mãi mới thấy . Nó gồm 2 bộ:
- Bộ 1: Hình ảnh có xe, có biển số xe ở trong và đã được đánh dấu vị trí.
- Bộ 2: Hình ảnh biển số đã cắt, và đã đánh dấu các kí tự của biển số xe
2. Train model : mình đã sử dụng model YoloV6 để train lấy weight (code trong bài). Ý tưởng ở đây là :
- Nhận diện biển số trong hình ảnh.
- Cắt hình ảnh đó ra.
- Đọc các kí tự trong hình ảnh.
- Ghi ra file excel.


Mình đã thực hiện thành công trên các bức ảnh và camera.
