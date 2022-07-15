#include <iostream>
template <typename T>
class Vector {
public:
    T* m_Buffer;
    size_t m_Tail;
    
public:
    Vector()
    {
        m_Buffer = new T(100);
        m_Tail = 0;
    };
    void PushBack(const T& number);
    ~Vector()
    {
        delete[] m_Buffer;
    }
    T& operator[](int index)
    {
        return m_Buffer[index];
    }
};

template<typename T>
void Vector<T>::PushBack(const T& number)
{
    m_Buffer[m_Tail] = number;
    m_Tail++;
}

int main()
{
    Vector<int> v;

    v.PushBack(10);
    v.PushBack(20);
    v.PushBack(30);

    std::cout << v[0] << ", " << v[1] << ", " << v[2] << std::endl;

    Vector<float> v2;
    v2.PushBack(0.1f);
    v2.PushBack(0.2f);
    v2.PushBack(0.3f);
    std::cout << v2[0] << ", " << v2[1] << ", " << v2[2] << std::endl;
    std::cin.get();
}