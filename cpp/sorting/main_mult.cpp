//
//  main.cpp
//  cpptest
//
//  Created by Alexander Ziskind on 1/11/21.
//

#include <iostream>
#include <vector>
#include <thread>
#include <algorithm> // for std::min
#include <cstdlib>   // for rand() and srand()
#include <ctime>     // for time()

void merge(int* arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary arrays
    std::vector<int> L(n1), R(n2);

    // Copy data to temp arrays L[] and R[]
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    // Merge the temp arrays back into arr[l..r]
    i = 0; j = 0; k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}


void iterativeMergeSortSingleThread(int* arr, int l, int r) {
    int n = r - l + 1;
    for (int curr_size = 1; curr_size <= n-1; curr_size = 2*curr_size) {
        for (int left_start = l; left_start < l + n-1; left_start += 2*curr_size) {
            int mid = std::min(left_start + curr_size - 1, l + n-1);
            int right_end = std::min(left_start + 2*curr_size - 1, l + n-1);
            merge(arr, left_start, mid, right_end);
        }
    }
}

void mergeSortParallel(int* arr, int n) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(num_threads);

    int chunk_size = n / num_threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        int left = i * chunk_size;
        int right = (i == num_threads - 1) ? (n - 1) : (left + chunk_size - 1);
        threads[i] = std::thread(iterativeMergeSortSingleThread, arr, left, right);
    }

    for (auto &t : threads) {
        t.join();
    }

    // Merge the sorted subarrays
    for (int size = chunk_size; size < n; size = 2*size) {
        for (int left = 0; left < n; left += 2*size) {
            int mid = left + size - 1;
            int right = std::min((left + 2*size - 1), (n-1));

            if (mid < right) {
                merge(arr, left, mid, right);
            }
        }
    }
}

void printArray(int arr[], int size) {
    // ... (printArray function remains the same)
}

int main() {
    //const int size = 1000000000; // Change this value as needed
    const int size = 10000000;
    int* arr = new int[size];

    srand(time(nullptr)); // Initialize random seed

    // Fill the array with random values
    for (int i = 0; i < size; i++)
        arr[i] = rand() % 100;

    // Sort the array using parallel merge sort
    mergeSortParallel(arr, size);

    // Print the sorted array
    // printArray(arr, size);

    delete[] arr;
    return 0;
}
