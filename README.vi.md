<p align="center">
  <br>
  <img src="./images/VanMach.svg" style="width:66%;">
</p>

# _VanMach-7B_ : Mô hình ngôn ngữ lớn Văn Mạch

### Ngôn ngữ tài liệu

[**Tiếng Anh**](./README.md) | [**Tiếng Trung giản thể**](./README.zh-CN.md) | [**Tiếng Trung phồn thể**](./README.zh-TW.md) | [**Tiếng Việt**](./README.vi.md)

## 1. Giới thiệu mô hình

**Mô hình ngôn ngữ lớn Văn Mạch (tiếng Trung: 文脉大模型; tiếng Anh: Van Mach LLM, mã mô hình: VanMach-7B)** là một mô hình ngôn ngữ lớn đa ngôn ngữ quy mô 7B, được phát triển hướng đến chữ viết truyền thống Việt Nam (Chữ Nôm), tư liệu lịch sử và các lĩnh vực ngôn ngữ, văn hóa. Mô hình được xây dựng trên nền tảng [SeaLLMs-v3-7B-Chat](https://huggingface.co/SeaLLMs/SeaLLMs-v3-7B-Chat), thông qua các bước mở rộng bộ từ vựng Chữ Nôm được thiết kế chuyên biệt, khởi tạo embedding từ bằng phương pháp tính toán toán học thuần túy, tiếp tục tiền huấn luyện (CPT) và tinh chỉnh có giám sát theo chỉ dẫn (SFT), nhằm tăng cường khả năng xử lý Chữ Nôm Việt Nam, tiếng Việt hiện đại, tư liệu lịch sử Hán Nôm và tri thức văn hóa truyền thống Việt Nam. Khác với các mô hình đa dụng chủ yếu hướng đến những tác vụ xử lý ngôn ngữ tự nhiên tiếng Việt hiện đại, VanMach-7B của chúng tôi lấy **biểu diễn tính toán ngôn ngữ học của Chữ Nôm, tri thức ngôn ngữ lịch sử và mối liên hệ giữa chúng với tiếng Việt hiện đại** làm định hướng cốt lõi trong quá trình phát triển mô hình. Mô hình hỗ trợ ba ngôn ngữ cùng các hình thức chữ viết tương ứng: **tiếng Việt** 【Chữ Quốc ngữ và Chữ Nôm được sử dụng trong lịch sử】, **tiếng Trung** 【chữ Hán giản thể và chữ Hán phồn thể】, **tiếng Anh**.

Chữ Nôm là một hệ thống chữ viết quan trọng từng được sử dụng để ghi chép tiếng Việt trong lịch sử. Hình thể của Chữ Nôm có mối liên hệ mật thiết với hệ thống chữ Hán, nhưng đồng thời mang những đặc điểm riêng về ý nghĩa, cách đọc, cấu tạo hình chữ và cách sử dụng thực tế. Nhiều chữ Nôm xuất hiện với tần suất cực kỳ thấp trong ngữ liệu huấn luyện của các mô hình ngôn ngữ lớn đa dụng hiện đại, thậm chí một số ký tự không thể có được biểu diễn Token độc lập và hoàn chỉnh thông qua các bộ tách Token hiện có. Trong khi đó, nguồn ngữ liệu huấn luyện Chữ Nôm đã được số hóa còn tương đối hạn chế, đặc biệt thiếu dữ liệu đối sánh với tiếng Việt hiện đại có quy mô lớn, chất lượng cao và được biên soạn một cách có hệ thống. Điều này khiến các phương pháp tinh chỉnh mô hình ngôn ngữ lớn truyền thống phải đối mặt với những thách thức đáng kể:

1. **Biểu diễn chữ viết chưa đầy đủ**: Nhiều chữ Nôm không có Token độc lập, dễ bị tách thành nhiều Token cấp từ con hoặc cấp byte, gây khó khăn cho việc hình thành biểu diễn ngữ nghĩa ổn định ở cấp độ từng chữ.
2. **Dữ liệu huấn luyện khan hiếm**: Tư liệu Hán Nôm và từ điển Chữ Nôm có giá trị quan trọng về ngôn ngữ học và lịch sử, nhưng quy mô dữ liệu có thể trực tiếp sử dụng để huấn luyện mô hình ngôn ngữ lớn còn tương đối hạn chế.
3. **Khởi tạo ngẫu nhiên hội tụ chậm**: Nếu trực tiếp khởi tạo ngẫu nhiên embedding cho các chữ Nôm mới được bổ sung, mô hình phải đồng thời học biểu diễn cơ bản của chữ viết, mối liên hệ về cách đọc và ngữ nghĩa theo ngữ cảnh từ nguồn ngữ liệu hạn chế.
4. **Khó khăn trong căn chỉnh giữa các hệ thống chữ viết**: Chữ Nôm, chữ Hán và Chữ Quốc ngữ tiếng Việt hiện đại có những mối quan hệ lịch sử và ngôn ngữ phức tạp, khó có thể thiết lập đầy đủ các mối liên hệ này nếu chỉ dựa vào việc huấn luyện trên văn bản thông thường.
5. **Hạn chế về tài nguyên tính toán**: Việc mở rộng bộ từ vựng ở quy mô lớn và tiếp tục huấn luyện thường đòi hỏi thêm bộ nhớ GPU và chi phí tính toán, đặc biệt bất lợi trong môi trường chỉ có một GPU.

Để giải quyết những vấn đề trên, VanMach-7B áp dụng một hướng tiếp cận kỹ thuật khác với việc chỉ khởi tạo ngẫu nhiên rồi trực tiếp tinh chỉnh:

**Trước hết, thông qua tri thức từ điển, các vector chữ Hán hiện có, biểu diễn cách đọc tiếng Việt và phương pháp tối ưu hóa lồi, mô hình xây dựng biểu diễn khởi tạo có cơ sở ngôn ngữ học cho các chữ Nôm mới được bổ sung trước khi bắt đầu huấn luyện (đưa tri thức vào embedding ở cấp độ từng chữ). Sau đó, thông qua CPT và SFT, mô hình từng bước học cách sử dụng thực tế của Chữ Nôm, các mối quan hệ tương ứng với tiếng Việt hiện đại và tri thức văn hóa liên quan.**

## 2. Thông tin cơ bản về mô hình

### 2.1 Thông số kỹ thuật của mô hình

| Hạng mục | Thông số |
|---|---|
| Tên mô hình | Văn Mạch (tiếng Trung: 文脉大模型; tiếng Anh: Van Mach) |
| Mã mô hình | VanMach-7B |
| Loại mô hình | Mô hình ngôn ngữ nhân quả tự hồi quy |
| Kiến trúc mô hình | Qwen2ForCausalLM |
| Mô hình nền tảng | SeaLLMs-v3-7B-Chat |
| Quy mô mô hình | **7.71B** (7,713,187,328) |
| Dung lượng dữ liệu trọng số mô hình | Khoảng **14.37 GiB** (15,426,374,656 byte) |
| Kích thước bộ từ vựng | 165,676 (kích thước ban đầu 152,064) |
| Số lượng Token Chữ Nôm mới được bổ sung | **13,612** |
| Số chiều embedding từ | 3,584 |
| Số tham số embedding từ và tầng đầu ra được bổ sung | Không chia sẻ trọng số, khoảng 97.57M (97,570,816) |
| Ngôn ngữ được hỗ trợ | Tiếng Việt, tiếng Trung, tiếng Anh |
| Các lĩnh vực được tăng cường | Chữ Nôm, tư liệu Hán Nôm, tiếng Việt hiện đại, lịch sử và văn hóa Việt Nam |

### 2.2 Nguồn gốc mô hình và hệ thống kế thừa kỹ thuật

VanMach-7B được phát triển trực tiếp trên cơ sở SeaLLMs-v3-7B-Chat.

<p align="center">
  <br>
  <img src="./images/vm_training_pipeline.svg" style="width:98%;">
</p>

SeaLLMs-v3-7B-Chat là một mô hình hội thoại đa ngôn ngữ hướng đến các ngôn ngữ Đông Nam Á. Theo bài báo kỹ thuật chính thức của SeaLLMs 3, mô hình nền tảng của nó có nguồn gốc từ **dòng mô hình Qwen2**.

## 3. Ngữ liệu huấn luyện và nguồn tri thức

Ngữ liệu huấn luyện của VanMach-7B tập trung vào chữ viết truyền thống Việt Nam, văn học lịch sử, từ điển Hán Nôm, dữ liệu nhận dạng chữ viết và tri thức từ vựng tiếng Việt hiện đại. Khác với các mô hình ngôn ngữ đa dụng chỉ dựa vào lượng lớn văn bản Internet để huấn luyện, dự án này tận dụng triệt để các từ điển và tư liệu chữ viết có cấu trúc ngôn ngữ học rõ ràng, sử dụng chúng cho quá trình khởi tạo embedding từ, tiếp tục tiền huấn luyện và tinh chỉnh có giám sát theo chỉ dẫn.

#### Tự Điển Chữ Nôm Dẫn Giải

《Tự Điển Chữ Nôm Dẫn Giải》（tiếng Trung: 喃字释义辞典） là một trong những nguồn tri thức ngôn ngữ Chữ Nôm quan trọng của dự án. Loại tư liệu này chứa các chữ Nôm cùng những thông tin liên quan về cách đọc, giải nghĩa và chữ viết, có giá trị quan trọng đối với việc thiết lập mối liên hệ giữa Chữ Nôm và tiếng Việt hiện đại. Trong quá trình phát triển mô hình, dữ liệu có tính chất từ điển chủ yếu được sử dụng để thiết lập quan hệ tương ứng giữa chữ Nôm và cách đọc bằng Chữ Quốc ngữ, cung cấp tri thức ngôn ngữ cần thiết cho việc khởi tạo toán học các Token Chữ Nôm mới được bổ sung, đồng thời xây dựng văn bản huấn luyện có cấu trúc.

#### Bảng tiêu chuẩn Chữ Nôm của Ủy ban Phục sinh Hán Nôm Việt Nam

Dự án cũng sử dụng bảng tiêu chuẩn Chữ Nôm do **Ủy ban Phục sinh Hán Nôm Việt Nam（tiếng Trung: 越南汉喃复兴委员会）** công bố. Các tài nguyên liên quan được sử dụng để bổ sung và hệ thống hóa thông tin về ký tự Chữ Nôm, cung cấp cơ sở tham khảo cho phạm vi bao phủ ký tự, mở rộng bộ từ vựng và xử lý chuẩn hóa chữ viết.

#### Các bộ dữ liệu Hán Nôm và âm Hán Việt trên Hugging Face

Ngoài ra, quá trình huấn luyện còn sử dụng một số bộ dữ liệu Hán Nôm và âm Hán Việt trên nền tảng Hugging Face. Những dữ liệu này được dùng để bổ sung các mối liên hệ về chữ viết và ngữ âm giữa Chữ Nôm, chữ Hán, âm Hán Việt và tiếng Việt.

#### Văn học Việt Nam cổ điển

Để mô hình có thể tiếp xúc với các hình thức ngôn ngữ trong bối cảnh lịch sử thực tế, VanMach-7B đưa vào nhiều tác phẩm văn học cổ điển Việt Nam.

Các tác phẩm chính bao gồm:

| Ngữ liệu | Nội dung và giá trị huấn luyện |
|---|---|
| _Truyện Kiều_（tiếng Trung: 金云翘传） | Tác phẩm tiêu biểu của văn học cổ điển Việt Nam, được sử dụng để xây dựng ngữ liệu song song Chữ Quốc ngữ - Chữ Nôm, bổ sung cách diễn đạt tiếng Việt cổ và từ vựng văn học |
| _Chinh phụ ngâm_（tiếng Trung: 征妇吟） | Được sử dụng để học ngôn ngữ văn học cổ điển, cách diễn đạt trong thơ ca và tri thức văn hóa liên quan |
| _Lục Vân Tiên_（tiếng Trung: 蓼云仙） | Được sử dụng để bổ sung văn học tự sự truyền thống Việt Nam và tư liệu ngôn ngữ lịch sử |
| _Tập thơ Hồ Xuân Hương_ | Được sử dụng để bổ sung thơ ca cổ điển Việt Nam, ngôn ngữ văn học, hình tượng văn hóa và cách diễn đạt truyền thống |
| _Đại Việt sử ký_（tiếng Trung: 大越史记） | Được sử dụng để bổ sung kiến thức về lịch sử Việt Nam, chế độ truyền thống, nhân vật, địa danh, sự kiện lịch sử, cũng như các hình thức chữ viết Hán văn và Hán Nôm liên quan. |

#### Dữ liệu văn bản OCR

Dự án sử dụng dữ liệu văn bản do **NomNaOCR** công bố.

## 4. Huấn luyện mô hình

Quá trình huấn luyện VanMach-7B được chia thành ba giai đoạn. Trước hết, từ điển Chữ Nôm, không gian embedding của mô hình gốc và phương pháp tối ưu hóa lồi được sử dụng để khởi tạo embedding đầu vào và trọng số LM Head cho 13,612 Token Chữ Nôm mới được bổ sung. Tiếp theo, thông qua tiếp tục tiền huấn luyện (CPT), mô hình kết hợp tư liệu Hán Nôm, tiếng Việt hiện đại và ngữ liệu từ điển có cấu trúc để tiếp tục huấn luyện biểu diễn của các chữ Nôm mới và những chữ Hán liên quan, đồng thời đưa vào tri thức ngôn ngữ và văn hóa chuyên ngành. Cuối cùng, tinh chỉnh có giám sát theo chỉ dẫn (SFT) được thực hiện bằng cách sử dụng ngữ liệu song song và dữ liệu hỏi đáp có cấu trúc, nhằm tăng cường khả năng hiểu và phản hồi chỉ dẫn của mô hình đối với Chữ Nôm, tư liệu Hán Nôm và tri thức ngôn ngữ, văn hóa Việt Nam.

### 4.1 Phương pháp khởi tạo embedding cho các chữ Nôm mới được bổ sung

Do mô hình gốc thiếu biểu diễn Token độc lập cho nhiều chữ Nôm, trong khi ngữ liệu huấn luyện Hán Nôm và tài nguyên tính toán có thể sử dụng đều tương đối hạn chế, việc trực tiếp áp dụng khởi tạo ngẫu nhiên khó có thể tận dụng đầy đủ tri thức ngôn ngữ học hiện có. Vì vậy, dự án này thiết kế một thuật toán khởi tạo embedding từ dựa trên ánh xạ từ điển, chuyển đổi hình chữ và tối ưu hóa lồi, nhằm xây dựng biểu diễn khởi tạo có cơ sở ngôn ngữ học cho các chữ Nôm mới được bổ sung trước khi bắt đầu huấn luyện bằng gradient. Giả sử kích thước bộ từ vựng của mô hình gốc là $V_0=152064$ , số chiều embedding từ là $d=3584$ , ma trận embedding đầu vào và ma trận trọng số đầu ra của mô hình ngôn ngữ lần lượt được ký hiệu là $E,W\in\mathbb{R}^{V_0\times d}$ . Đối với một Token ID bất kỳ $t$ trong bộ tách Token gốc, ký hiệu $E_t$ và $W_t$ lần lượt là các vector hàng tương ứng trong hai ma trận này. Thuật toán xây dựng các vector khởi tạo cho ký tự mới một cách riêng biệt trong không gian tham số đầu vào và đầu ra.

**Nhận dạng ký tự và ánh xạ trực tiếp.** Trước hết, thuật toán đọc từ điển ánh xạ giữa Chữ Nôm và cách đọc bằng Chữ Quốc ngữ, thực hiện chuẩn hóa Unicode NFC đối với ký tự và cách đọc, chuẩn hóa chữ hoa, chữ thường và khoảng trắng, đồng thời hợp nhất các cách đọc trùng lặp. Sau đó, bộ tách Token gốc được sử dụng để nhận dạng các ký tự đã biết: chỉ khi một ký tự có thể được mã hóa thành đúng một Token không đặc biệt và kết quả giải mã hoàn toàn trùng khớp với ký tự ban đầu thì ký tự đó mới được coi là có biểu diễn độc lập hợp lệ. Đối với những chữ Nôm chưa thể được mã hóa độc lập, thuật toán ưu tiên thử chuyển đổi hình chữ bằng quy tắc `t2s` của OpenCC; nếu kết quả chuyển đổi là một ký tự đơn khác với ký tự ban đầu và đã có Token độc lập, trọng số của ký tự đó trong ma trận embedding đầu vào và LM Head sẽ được sao chép riêng biệt. Các ký tự còn lại được chuyển sang quy trình khởi tạo toán học dựa trên cách đọc.

<p align="center">
  <br>
  <img src="./images/vm_semantic_geometry.svg" style="width:85%;">
</p>

**Biểu diễn cách đọc và các điểm neo tham chiếu.** Để tận dụng mối quan hệ tương ứng giữa các ký tự hiện có và cách đọc tiếng Việt hiện đại, trước hết thuật toán xây dựng tập tham chiếu cách đọc dựa trên từ điển. Cụ thể, các ký tự đã biết có cùng cách đọc bằng Chữ Quốc ngữ trong từ điển được nhóm lại với nhau, đồng thời một nhóm vector tham chiếu được thiết lập cho mỗi cách đọc hợp lệ cùng tập ký tự đã biết tương ứng. Giả sử $q$ biểu thị một cách đọc bằng Chữ Quốc ngữ, $T(q)=(t_1,\ldots,t_m)$ là chuỗi Token ID thu được khi bộ tách Token gốc mã hóa cách đọc đó, trong đó $m=|T(q)|$ là độ dài chuỗi; $H(q)$ là tập hợp Token ID tương ứng với những ký tự đã biết trong từ điển có cùng cách đọc đó và có thể được bộ tách Token gốc mã hóa độc lập. Đối với các cách đọc có $T(q)$ hợp lệ và $H(q)$ không rỗng, định nghĩa:

$$
v_q=\frac{1}{|T(q)|}\sum_{t\in T(q)}E_t,\qquad
C_q=\frac{1}{|H(q)|}\sum_{h\in H(q)}E_h,\qquad
U_q=\frac{1}{|H(q)|}\sum_{h\in H(q)}W_h
$$

Trong đó, $v_q$ biểu thị vị trí của cách đọc bằng Chữ Quốc ngữ trong không gian embedding gốc, còn $C_q$ và $U_q$ lần lượt biểu thị tâm embedding đầu vào và tâm trọng số LM Head của các ký tự đã biết có cùng cách đọc. Như vậy, mỗi cách đọc hợp lệ tương ứng với một quan hệ tham chiếu bộ ba được tạo thành từ $(v_q,C_q,U_q)$ , gọi là **điểm neo cách đọc** . Điểm neo cách đọc sử dụng cách đọc bằng Chữ Quốc ngữ làm chỉ mục, thông qua quan hệ tương ứng giữa chữ và âm trong từ điển để liên kết biểu diễn vector của cách đọc với các tham số đầu vào và đầu ra của những ký tự hiện có. Vector cách đọc được tạo ra bằng cách lấy trung bình embedding của kết quả tách Token gốc mà không cần huấn luyện thêm bộ mã hóa; đối với những cách đọc chứa Token không xác định hoặc Token đặc biệt, hoặc không thể mã hóa hợp lệ, thuật toán không xây dựng điểm neo cách đọc hợp lệ. Các ký tự tham chiếu được giới hạn nghiêm ngặt trong những ký tự thuộc từ điển đã có thể được bộ tách Token gốc mã hóa độc lập, thay vì trực tiếp sử dụng toàn bộ vùng chữ Hán Unicode hoặc tất cả chữ Hán trong bộ từ vựng gốc làm điểm neo.

**Tối ưu hóa lồi và chuyển giao vector.** Đối với cách đọc mục tiêu không thể trực tiếp thiết lập tâm tham chiếu từ các ký tự đồng âm nhưng có thể tính được vector cách đọc thông qua bộ tách Token gốc, thuật toán chuyển bài toán biểu diễn thành bài toán chiếu vector cách đọc lên bao lồi của các điểm neo cách đọc hiện có. Giả sử tập tham chiếu cách đọc chứa $n$ điểm neo hợp lệ, có các cách đọc bằng Chữ Quốc ngữ tương ứng lần lượt là $q_1,q_2,\ldots,q_n$ , đồng thời ký hiệu ngắn gọn ba vector của mỗi điểm neo là $v_i=v_{q_i}$ , $C_i=C_{q_i}$ và $U_i=U_{q_i}$ . Đối với cách đọc mục tiêu $q$ không nằm trong tập tham chiếu này, trước hết thuật toán tính vector cách đọc $v_q$ theo phương pháp đã trình bày, sau đó tìm một tập hệ số tổ hợp $\boldsymbol{\alpha}=(\alpha_1,\ldots,\alpha_n)$ sao cho khoảng cách Euclid giữa tổng có trọng số của các vector cách đọc tham chiếu và vector cách đọc mục tiêu đạt giá trị nhỏ nhất:

$$
\boldsymbol{\alpha}^{*}
=\arg\min_{\boldsymbol{\alpha}}
\frac{1}{2}\left\|
\sum_{i=1}^{n}\alpha_i v_i-v_q
\right\|_2^2,
\qquad
\alpha_i\geq0,\quad
\sum_{i=1}^{n}\alpha_i=1
$$

Trong đó, $\alpha_i$ biểu thị trọng số tổ hợp của điểm neo cách đọc thứ $i$ , $\boldsymbol{\alpha}^{*}$ biểu thị vector hệ số tối ưu thỏa mãn các điều kiện ràng buộc, còn $\|\cdot\|_2$ biểu thị chuẩn Euclid. Đây là một bài toán tối ưu hóa bậc hai lồi với ràng buộc đơn hình xác suất, có mục tiêu tìm vị trí gần nhất với cách đọc mục tiêu trong bao lồi được tạo bởi các vector cách đọc tham chiếu hiện có. Sau khi giải bài toán, thuật toán không trực tiếp sử dụng vector hình chiếu lồi làm embedding ký tự, mà áp dụng các hệ số tổ hợp thu được lần lượt lên các tâm tham chiếu đầu vào và đầu ra của những điểm neo cách đọc tương ứng:

```math
\widehat{E}_q=\sum_{i=1}^{n}\alpha_i^{*}C_i \qquad \widehat{W}_q=\sum_{i=1}^{n}\alpha_i^{*}U_i
```

Trong đó, $\widehat E_q$ và $\widehat W_q$ lần lượt biểu thị vector tham chiếu embedding đầu vào và vector tham chiếu trọng số LM Head được xây dựng dựa trên cách đọc mục tiêu $q$ . Do đó, vector cách đọc bằng Chữ Quốc ngữ đảm nhiệm việc xác định quan hệ tổ hợp giữa các điểm neo tham chiếu khác nhau, trong khi embedding đầu vào và trọng số LM Head của những ký tự hiện có được sử dụng để xây dựng các tham số khởi tạo thực tế. Điều này cho phép thuật toán tận dụng biểu diễn cách đọc tiếng Việt cùng thông tin tham số chữ Hán và Chữ Nôm đã có trong mô hình gốc, thay vì đặt tất cả ký tự mới vào những vị trí ngẫu nhiên.

**Giải số và kiểm soát hội tụ.** Trong quá trình triển khai thực tế, thuật toán sử dụng phương pháp lặp gradient có điều kiện kiểu Frank–Wolfe, lựa chọn đỉnh tham chiếu dựa trên phần dư hình chiếu hiện tại và cập nhật tổ hợp lồi thông qua phép tìm kiếm bước giải tích bị ràng buộc trong khoảng $[0,1]$ . Thuật toán không xây dựng toàn bộ ma trận Gram của tập điểm neo quy mô lớn, mà sử dụng GPU để thực hiện tích vô hướng vector và các phép toán ma trận ở độ chính xác FP64, đồng thời định kỳ sử dụng SLSQP để hiệu chỉnh cục bộ tập tham chiếu đang hoạt động. Các vector tham chiếu được chuẩn hóa tỷ lệ theo trung vị chuẩn của chúng. Số vòng lặp tối đa mặc định là 1,800, dung sai tối ưu là $2\times10^{-8}$ , đồng thời khoảng cách tối ưu Frank–Wolfe được sử dụng để đánh giá mức độ hội tụ số của quá trình tối ưu hóa chính.

Do các tổ hợp lồi khác nhau có thể tạo ra cùng một vector hình chiếu, sau khi quá trình tối ưu hóa chính đáp ứng điều kiện hội tụ, thuật toán tiếp tục thử thực hiện tối ưu hóa chuẩn nhỏ nhất. Giả sử vị trí hình chiếu thu được từ quá trình tối ưu hóa chính là $p^{*}=\sum_{i=1}^{n}\alpha_i^{*}v_i$ , khi giữ nguyên vị trí hình chiếu này, thuật toán tiếp tục giải bài toán:

$$
\min_{\boldsymbol{\alpha}}
\frac{1}{2}\|\boldsymbol{\alpha}\|_2^2,
\qquad
\sum_{i=1}^{n}\alpha_i v_i=p^{*},\quad
\sum_{i=1}^{n}\alpha_i=1,\quad
\alpha_i\geq0
$$

Bước tối ưu hóa thứ cấp này nhằm lựa chọn nghiệm có chuẩn Euclid nhỏ hơn trong số các tổ hợp hệ số khả thi tương ứng với cùng một vị trí hình chiếu. Nếu nghiệm vượt qua quá trình kiểm tra số, các hệ số đã hiệu chỉnh sẽ được sử dụng để xây dựng các vector tham chiếu đầu vào và đầu ra nêu trên. Quá trình tối ưu hóa thứ cấp kết hợp phân rã giá trị kỳ dị (SVD), SLSQP, tối ưu hóa đối ngẫu L-BFGS-B và kiểm tra các điều kiện tối ưu Karush–Kuhn–Tucker (KKT), nhằm tìm kiếm nghiệm hệ số ổn định trong khi vẫn giữ nguyên vị trí hình chiếu và các ràng buộc tổ hợp lồi. Đối với các trường hợp chưa đạt hội tụ hoàn toàn hoặc không vượt qua kiểm tra số, chương trình giữ lại tổ hợp khả thi tốt hơn đã tìm được và ghi nhận trạng thái tương ứng, tránh việc trực tiếp làm mất kết quả khởi tạo ký tự chỉ vì quá trình giải số cục bộ thất bại.

**Xử lý nhiều cách đọc và cơ chế dự phòng.** Đối với các chữ Nôm có nhiều cách đọc bằng Chữ Quốc ngữ, thuật toán xây dựng riêng các vector tham chiếu đầu vào và đầu ra tương ứng với từng cách đọc, sau đó lấy trung bình số học của kết quả từ các cách đọc để thu được tham số khởi tạo của ký tự đó. Cụ thể, giả sử $c$ là chữ Nôm cần khởi tạo, $R(c)$ là tập hợp các cách đọc bằng Chữ Quốc ngữ khác nhau được ghi nhận cho ký tự này trong từ điển, còn $e_c^{(0)}$ và $w_c^{(0)}$ lần lượt là vector embedding đầu vào và vector LM Head khởi tạo cuối cùng của ký tự đó. Đối với những ký tự được khởi tạo dựa trên cách đọc, ta có:

$$
e_c^{(0)}=\frac{1}{|R(c)|}\sum_{q\in R(c)}\widehat E_q,
\qquad
w_c^{(0)}=\frac{1}{|R(c)|}\sum_{q\in R(c)}\widehat W_q
$$

Trong đó, đối với cách đọc có thể khớp trực tiếp với tập tham chiếu cách đọc, $\widehat E_q$ và $\widehat W_q$ được lấy trực tiếp từ $C_q$ và $U_q$ tương ứng; đối với cách đọc không khớp trực tiếp nhưng vẫn có thể được mã hóa hợp lệ, thuật toán sử dụng kết quả được xây dựng thông qua tối ưu hóa lồi. Nếu một cách đọc riêng lẻ không thể được mã hóa hợp lệ, thiếu điểm neo tham chiếu khả dụng hoặc không thể thiết lập bài toán tối ưu hóa hợp lệ, vector embedding đầu vào trung bình và vector trọng số LM Head trung bình của tất cả ký tự đã biết trong từ điển sẽ được sử dụng làm vector dự phòng cho cách đọc đó; các kết quả hợp lệ đã có từ những cách đọc khác vẫn được giữ lại và tham gia vào phép tính trung bình cuối cùng. Vì vậy, cơ chế dự phòng của thuật toán được thực hiện ở cấp độ cách đọc, thay vì loại bỏ toàn bộ thông tin đã biết của một ký tự chỉ vì thiếu một cách đọc.

**Phân biệt các vector trùng lặp.** Đối với các ký tự mới được bổ sung có cùng tổ hợp cách đọc hoặc quan hệ tham chiếu ( **các dị thể Chữ Nôm** ), quá trình khởi tạo có thể tạo ra những vector giống nhau. Để giảm thiểu tình trạng này, chương trình kiểm tra riêng biểu diễn vector của embedding đầu vào và LM Head ở độ chính xác lưu trữ thực tế, đồng thời thực hiện nhiễu loạn hiệu quả nhỏ nhất khi cần thiết. Cụ thể, thuật toán lấy mẫu ngẫu nhiên một hướng đơn vị từ phân phối Gauss, dựa trên các giá trị có thể biểu diễn liền kề và ranh giới làm tròn của những định dạng trọng số thực tế như FP16, BF16 hoặc FP32 để tính toán kích thước bước nhỏ nhất xấp xỉ có khả năng tạo ra thay đổi số học có hiệu lực, sau đó sử dụng tìm kiếm nhị phân để tiếp tục thu nhỏ biên độ nhiễu loạn; khi cần thiết, phép điều chỉnh một ULP (Unit in the Last Place, tức đơn vị ở vị trí chữ số cuối cùng) trên một tọa độ đơn lẻ được sử dụng làm phương án dự phòng. Quá trình này tuân theo nguyên tắc bảo toàn vector khởi tạo ban đầu ở mức tối đa có thể, đồng thời giữ nguyên các quan hệ sao chép vector hợp lệ được thiết lập thông qua ánh xạ trực tiếp hình chữ.

Giai đoạn này được hoàn thành hoàn toàn dựa trên các trọng số hiện có, tri thức từ điển và tính toán số, không thực hiện huấn luyện. Vai trò chủ yếu của nó là chuyển đổi các mối quan hệ biểu diễn hiện có giữa chữ Hán, Chữ Nôm và tiếng Việt hiện đại thành các tham số khởi tạo cho ký tự mới, giúp giai đoạn CPT tiếp theo có thể bắt đầu từ các vector có cơ sở ngôn ngữ học, rồi tiếp tục học cách đọc, cách sử dụng và ngữ nghĩa của ký tự thông qua ngữ cảnh thực tế.

### 4.2 Tiếp tục tiền huấn luyện (CPT)

Giai đoạn CPT chủ yếu hướng đến việc học biểu diễn chữ Hán Nôm, mô hình hóa ngôn ngữ tiếng Việt và đưa tri thức chuyên ngành vào mô hình. Ngữ liệu huấn luyện bao gồm các tư liệu văn học và lịch sử như 《Truyện Kiều》《Chinh phụ ngâm》《Lục Vân Tiên》《Đại Việt sử ký》《Tập thơ Hồ Xuân Hương》, dữ liệu văn bản NomNaOCR, cùng các từ điển Chữ Nôm, bảng tiêu chuẩn Chữ Nôm, bộ dữ liệu âm Hán Việt và từ điển tiếng Việt hiện đại. Đối với các văn bản liên tục như tác phẩm văn học và tư liệu lịch sử, quá trình huấn luyện mô hình ngôn ngữ tự hồi quy được thực hiện trực tiếp; đối với từ điển, bảng cách đọc và các tài liệu có cấu trúc khác, dữ liệu được chuyển đổi thành văn bản liên tục theo các mẫu định nghĩa trước, nhằm thiết lập mối liên hệ giữa ký tự Chữ Nôm, cách đọc bằng Chữ Quốc ngữ, âm Hán Việt, giải nghĩa và ngữ cảnh.

Giai đoạn này sử dụng thuật toán QLoRA, với trọng số mô hình nền tảng được lượng tử hóa bốn bit theo định dạng NF4, hạng LoRA được đặt là $r=64$ , và độ dài cắt ngắn văn bản là 1,024 Token. Khác với phương pháp thông thường chỉ huấn luyện các bộ thích ứng hạng thấp, giai đoạn CPT còn bỏ trạng thái đóng băng đối với các embedding ký tự liên quan và LM Head, cho phép embedding ở cấp độ từng chữ của các chữ Hán giản thể, chữ Hán phồn thể và Chữ Nôm được cập nhật gradient một cách trực tiếp, đồng thời cho phép LM Head tham gia huấn luyện. Cách thiết lập này vừa có thể tiếp tục hiệu chỉnh các vector Chữ Nôm mới được khởi tạo bằng phương pháp toán học ở giai đoạn đầu tiên, vừa có thể điều chỉnh embedding của các chữ Hán đồng thời được sử dụng làm chữ Nôm.

### 4.3 Tinh chỉnh có giám sát theo chỉ dẫn (SFT)

Giai đoạn SFT chủ yếu sử dụng ngữ liệu song song, từ điển và các dữ liệu có cấu trúc khác để xây dựng các mẫu hỏi đáp có giám sát, nhằm tăng cường khả năng vận dụng tri thức của mô hình về cách đọc Chữ Nôm, giải nghĩa, âm Hán Việt, quan hệ tương ứng với Chữ Quốc ngữ và lịch sử, văn hóa Việt Nam, đồng thời tiếp tục huấn luyện khả năng giải thích văn bản Hán Nôm và phản hồi các chỉ dẫn xuyên ngôn ngữ. Quá trình huấn luyện sử dụng thuật toán LoRA với độ chính xác FP8, hạng được đặt là $r=32$ , và độ dài cắt ngắn văn bản là 1,536 Token.

Khác với giai đoạn CPT, giai đoạn SFT đóng băng tầng embedding đầu vào và LM Head, chỉ cập nhật tham số thông qua các bộ thích ứng hạng thấp. Thiết lập này cho phép mô hình duy trì các biểu diễn ký tự đã được hình thành trong hai giai đoạn trước, đồng thời chuyển trọng tâm huấn luyện sang hỏi đáp tri thức có cấu trúc, hiểu chỉ dẫn và sinh câu trả lời.

|  | Thuật toán | Hạng | Độ chính xác lượng tử hóa | Độ dài cắt ngắn |
| --- | --- | --- | --- | --- |
| CPT | QLoRA | 64 | NF4 | 1,024 |
| SFT | LoRA | 32 | FP8 | 1,536 |

### 4.4 Môi trường phần cứng và phần mềm

VanMach-7B được tinh chỉnh bằng một GPU NVIDIA GeForce RTX 5060 Ti 16GB duy nhất, sử dụng framework huấn luyện Axolotl v0.20.0. Hai giai đoạn CPT và SFT lần lượt sử dụng QLoRA và LoRA nhằm thích ứng với giới hạn tài nguyên tính toán của môi trường một GPU có bộ nhớ đồ họa 16GB.

## 5. Ví dụ sử dụng

VanMach-7B được xây dựng trên hệ sinh thái Hugging Face Transformers và có thể được tải bằng phương pháp tiêu chuẩn dành cho mô hình ngôn ngữ tự hồi quy.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "./VanMach-7B"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype=torch.bfloat16, device_map="cuda"
)
messages = []

while True:
    question = input("User: ")
    messages.append({"role": "user", "content": question})
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=512)
    answer = tokenizer.decode(
        outputs[0][inputs.shape[-1]:], skip_special_tokens=True
    )
    print("Assistant:", answer)
    messages.append({"role": "assistant", "content": answer})
```

## 6. Nguồn gốc, giấy phép và tuyên bố sử dụng

### 6.1 Giấy phép và phân phối lại

SeaLLMs-v3-7B-Chat sử dụng giấy phép tùy chỉnh của SeaLLMs. Là một mô hình phái sinh từ mô hình này, VanMach-7B phải tuân thủ các điều kiện sử dụng và phân phối lại hiện hành áp dụng cho mô hình gốc. Các điều khoản liên quan bao gồm tuyên bố quyền sở hữu ban đầu, phân phối lại mô hình phái sinh, thông báo sửa đổi, ghi nhận nguồn gốc và tác giả, hạn chế sử dụng cùng các điều kiện khác. Khi công bố và sử dụng mô hình này, các yêu cầu cấp phép áp dụng từ mô hình gốc cũng phải được giữ nguyên và tuân thủ.

Các mô tả kỹ thuật trong thẻ mô hình này không cấu thành bất kỳ sự sửa đổi hoặc miễn trừ nào đối với những hạn chế cấp phép của mô hình nền tảng.

### 6.2 Tuyên bố liên quan đến dữ liệu huấn luyện

Các tài liệu được VanMach-7B sử dụng bao gồm văn học truyền thống, văn bản lịch sử, từ điển, dữ liệu OCR và các bộ dữ liệu ngôn ngữ liên quan trên những nền tảng công khai. Các nguồn dữ liệu khác nhau có thể chịu sự điều chỉnh của những điều kiện khác nhau về bản quyền, sử dụng và phân phối lại. Việc công bố rộng rãi mô hình không đồng nghĩa với việc tất cả tài liệu huấn luyện đều có thể được phân phối lại mà không bị hạn chế. Việc sử dụng và phân phối lại ngữ liệu huấn luyện phải tuân thủ các giấy phép và yêu cầu pháp luật tương ứng áp dụng cho từng nguồn dữ liệu.

### 6.3 Tuyên bố miễn trừ trách nhiệm

VanMach-7B chủ yếu được sử dụng trong xử lý ngôn ngữ tự nhiên, nghiên cứu chữ viết Hán Nôm và các ứng dụng ngôn ngữ, văn hóa liên quan. Do mô hình sử dụng các phương pháp học thống kê, kết quả đầu ra có thể chứa sai sót về sự kiện, nhầm lẫn cách đọc, giải nghĩa không chính xác hoặc những nội dung thiếu chính xác khác. Trong các lĩnh vực giáo dục, nghiên cứu tư liệu lịch sử, dịch thuật chuyên nghiệp và những trường hợp khác đòi hỏi độ chính xác cao, kết quả do mô hình tạo ra cần được sử dụng kết hợp với tài liệu tham khảo đáng tin cậy và quá trình kiểm tra của con người.

## 7. Tác giả

**Học Viện Điện Tử Và Công Nghệ Thông Tin, Đại Học Giao thông Lan Châu** [**ĐỖ Vũ**](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1837.htm)（tiếng Trung: 杜宇, tiếng Anh: DU Yu, No.11250717, email: <qluduyu09@163.com> hoặc <qluduyu09@gmail.com> hoặc <11250717@stu.lzjtu.edu.cn> ）

