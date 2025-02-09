import re

def extract_boxed_content(latex_string):
    # 使用正则表达式匹配 \boxed{} 中的内容
    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, latex_string)
    return matches[0]

# 示例字符串
latex_string = "Here is an example: \\boxed{B} and another one \\boxed{x^2 + y^2 = z^2}."
latex_string = "To find the degree of the field extension \\( \\mathbb{Q}(\\sqrt{2}, \\sqrt{3}, \\sqrt{18}) \\) over \\( \\mathbb{Q} \\):\n\n1. First, simplify \\( \\sqrt{18} \\). Since \\( \\sqrt{18} = 3\\sqrt{2} \\), it is already contained in \\( \\mathbb{Q}(\\sqrt{2}) \\).\n2. Therefore, the field \\( \\mathbb{Q}(\\sqrt{2}, \\sqrt{3}, \\sqrt{18}) \\) is the same as \\( \\mathbb{Q}(\\sqrt{2}, \\sqrt{3}) \\).\n3. To determine the degree of the extension \\( \\mathbb{Q}(\\sqrt{2}, \\sqrt{3}) \\) over \\( \\mathbb{Q} \\), we note that both \\( \\sqrt{2} \\) and \\( \\sqrt{3} \\) have minimal polynomials \\( x^2 - 2 \\) and \\( x^2 - 3 \\) respectively, which are irreducible over \\( \\mathbb{Q} \\).\n4. The degree of the extension \\( \\mathbb{Q}(\\sqrt{2}, \\sqrt{3}) \\) over \\( \\mathbb{Q} \\) is the product of the degrees of each extension:\n   - \\( [\\mathbb{Q}(\\sqrt{2}) : \\mathbb{Q}] = 2 \\)\n   - \\( [\\mathbb{Q}(\\sqrt{2}, \\sqrt{3}) : \\mathbb{Q}(\\sqrt{2})] = 2 \\)\n   - Therefore, \\( [\\mathbb{Q}(\\sqrt{2}, \\sqrt{3}) : \\mathbb{Q}] = 2 \\times 2 = 4 \\)\n\nThus, the degree of the field extension \\( \\mathbb{Q}(\\sqrt{2}, \\sqrt{3}, \\sqrt{18}) \\) over \\( \\mathbb{Q} \\) is 4.\n\n\\[\n\\boxed{B}\n\\]"
# 提取框内内容
boxed_contents = extract_boxed_content(latex_string)

# 打印结果
print("Extracted contents:", boxed_contents)