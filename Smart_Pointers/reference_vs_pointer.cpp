// Ponter has the freedom to move around and point to different variables
// whereas a reference is assigned onetime and it just becomes a reference to
// that location in memory.

#include <iostream>

using namespace std;

void swap_by_ptr(int *ptr1, int *ptr2) {
  int s = *ptr1;
  *ptr1 = *ptr2;
  *ptr2 = s;
}

struct demo {
  int a;
};

int main() {
  int x = 5;
  int y = 6;
  demo d;

  int *p;
  // output's : Value of p before any assignment : 0x5589a3537a90
  // cout << "Value of p before any assignment : " << p << endl;
  p = &x; // p stores address of variable x
  cout << "Value of p : " << p << endl;
  cout << "Value of *p, x : " << *p << " " << x
       << endl; // accessing the value at address stored in p
  *p = 11;      // value of x becomes 11
  cout << "Value of x : " << x << endl;
  int q = *p;
  // int a = p; // Cannot initialize a variable of type 'int' with an lvalue of
  // type 'int *'
  cout << "Value of q : " << q << endl;
  p = &y; // Pointer reinitialization allowed
  cout << "Value of *p after reinitialization : " << p << endl;
  cout << "Value of *p after reinitialization : " << *p << endl;
  cout << endl;

  int &r = x; // reference to memory location of x
  // &r = y;  // Compile error
  cout << "Value of x, y, r : " << x << " " << y << " " << r << endl;
  cout << "Address of x, r : " << &x << " " << &r << endl;
  r = y; // value of x is now equal to y (i.e. 6) because x and r are the same
         // location
  cout << "Value of x, y, r : " << x << " " << y << " " << r << endl;
  cout << &r << endl;

  p++; // points to the next memory location
  r++; // value of x is incremented (i.ei becomes 7)
  cout << "Value of p : " << p << " " << *p << endl;
  cout << "Value of r, x : " << r << " " << x << endl;
  cout << endl;

  d.a = 10;
  demo *z = &d;
  demo &zz = d;
  cout << z->a << endl;
  cout << zz.a << endl;
  cout << d.a << endl;
  z->a = 20;
  zz.a = 20;
  cout << d.a << endl;
  cout << z->a++ << endl;
  cout << z++->a++ << endl;
  cout << zz.a++ << endl;
  cout << d.a << endl;
  cout << endl;

  int *ptr1 = &x;
  int *ptr2 = &y;
  cout << *ptr1 << " " << *ptr2 << endl;
  swap_by_ptr(ptr1, ptr2);
  cout << *ptr1 << " " << *ptr2 << endl;

  return 0;
}