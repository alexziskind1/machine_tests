import sys
import random

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2  # Finding the mid of the array
        L = arr[:mid]        # Dividing the array elements into 2 halves
        R = arr[mid:]

        merge_sort(L)  # Sorting the first half
        merge_sort(R)  # Sorting the second half

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

if __name__ == "__main__":
    # Handling command line argument for array size
    if len(sys.argv) != 2:
        print("Usage: python mergesort.py <size_of_array>")
        sys.exit(1)

    try:
        size = int(sys.argv[1])
    except ValueError:
        print("Please enter a valid integer for the array size.")
        sys.exit(1)

    # Generating a random array of given size
    array = [random.randint(0, size) for _ in range(size)]
    print("Original array:", array)

    merge_sort(array)
    
    print("")
    print("Sorted array:", array)
