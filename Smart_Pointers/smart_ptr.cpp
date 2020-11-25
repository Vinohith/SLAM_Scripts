#include <iostream>
#include <memory>

using namespace std;

class Rectangle {
  int length;
  int width;

public:
  Rectangle(int l, int w) : length(l), width(w) {}
  const int area() { return length * width; }
};

int main() {
  int a=12;
  int b;
  cout << a << endl;
  b = a++;
  cout << a << "  " << b << endl;
  a=12;
  b = ++a;
  cout << a << "  " << b << endl;

  cout << "Unique Pointer" << endl;
  unique_ptr<Rectangle> P1(new Rectangle(10, 5));
  cout << P1->area() << endl;
  unique_ptr<Rectangle> P2;
  P2 = move(P1);
  cout << P2->area() << endl;
  //   This will give error as the pointer is moved.
  //   cout << P1->area() << endl;
  cout << "**********************" << endl;

  cout << "Shared Pointer" << endl;
  shared_ptr<Rectangle> P3(new Rectangle(10, 5));
  cout << P3->area() << endl;
  shared_ptr<Rectangle> P4;
  P4 = P3;
  cout << P4->area() << endl;
  //   This will now not give error
  cout << P3->area() << endl;
  // prints 2 as the Reference counter is 2
  cout << P3.use_count() << endl;
  cout << P4.use_count() << endl;
  cout << "**********************" << endl;

  cout << "Weak Pointer" << endl;
  weak_ptr<Rectangle> P5;
  P5 = P3;
  cout << P5.lock()->area() << endl;

  cout << "**********************" << endl;

  return 0;
}