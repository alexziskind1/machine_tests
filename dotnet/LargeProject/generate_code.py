import os

# Generate 100,000 namespaces and classes
for i in range(1, 100001):  # Adjust this number as needed
    directory = f"GeneratedCode/Namespace{i}"
    os.makedirs(directory, exist_ok=True)
    with open(f"{directory}/Class{i}.cs", "w") as f:
        f.write(
            f"""
        namespace Namespace{i}
        {{
            public class Class{i}
            {{
                public int Value => {i};

                public int ComplexCalculation()
                {{
                    return RecursiveCalculation(10) + NestedLoopCalculation() + ArrayProcessing();
                }}

                private int RecursiveCalculation(int depth)
                {{
                    if (depth == 0)
                        return Value;
                    return Value + RecursiveCalculation(depth - 1);
                }}

                private int NestedLoopCalculation()
                {{
                    int result = 0;
                    for (int j = 0; j < 100; j++)  // Outer loop
                    {{
                        for (int k = 0; k < 100; k++)  // Inner loop
                        {{
                            result += j * k * Value;
                        }}
                    }}
                    return result;
                }}

                private int ArrayProcessing()
                {{
                    int[] data = new int[10000];
                    for (int i = 0; i < data.Length; i++)
                    {{
                        data[i] = i * Value;
                    }}

                    int result = 0;
                    foreach (var item in data)
                    {{
                        result += item;
                    }}

                    return result;
                }}
            }}
        }}
        """
        )
