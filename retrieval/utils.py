import numpy as np

def cosine_similarity(a: np.array, b: np.array):
    dot = np.dot(a, b)
    norm1 = np.linalg.norm(a)
    norm2 = np.linalg.norm(b)
    return dot/ (norm1* norm2)

if __name__ == "__main__":
    a = np.array([1, 2, 3])
    b = np.array([-1, -2, -3])
    print(f"Cosine similarity{a, b}: \n {cosine_similarity(a, b)}")
