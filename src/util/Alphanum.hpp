#ifndef ALPHANUM__HPP
#define ALPHANUM__HPP

/*
 Released under the MIT License - https://opensource.org/licenses/MIT

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/* $Header: /code/doj/alphanum.hpp,v 1.3 2008/01/28 23:06:47 doj Exp $ */

#include <cassert>
#include <functional>
#include <string>
#include <sstream>

#ifdef ALPHANUM_LOCALE
#include <cctype>
#endif

#ifdef DOJDEBUG
#include <iostream>
#include <typeinfo>
#endif

// TODO: make comparison with hexadecimal numbers. Extend the alphanum_comp() function by traits to choose between decimal and hexadecimal.

namespace doj
{

  // anonymous namespace for functions we use internally. But if you
  // are coding in C, you can use alphanum_impl() directly, since it
  // uses not C++ features.
  namespace {

    // if you want to honour the locale settings for detecting digit
    // characters, you should define ALPHANUM_LOCALE
#ifdef ALPHANUM_LOCALE
    /** wrapper function for ::isdigit() */
    bool alphanum_isdigit(int c)
    {
      return isdigit(c);
    }
#else
    /** this function does not consider the current locale and only
	works with ASCII digits.
	@return true if c is a digit character
    */
    bool alphanum_isdigit(const char c)
    {
      return c>='0' && c<='9';
    }
#endif

    /**
       compare l and r with strcmp() semantics, but using
       the "Alphanum Algorithm". This function is designed to read
       through the l and r strings only one time, for
       maximum performance. It does not allocate memory for
       substrings. It can either use the C-library functions isdigit()
       and atoi() to honour your locale settings, when recognizing
       digit characters when you "#define ALPHANUM_LOCALE=1" or use
       it's own digit character handling which only works with ASCII
       digit characters, but provides better performance.

       @param l NULL-terminated C-style string
       @param r NULL-terminated C-style string
       @return negative if l<r, 0 if l equals r, positive if l>r
     */
    int alphanum_impl(const char *l, const char *r)
    {
      enum mode_t { STRING, NUMBER } mode=STRING;

      while(*l && *r)
	{
	  if(mode == STRING)
	    {
	      char l_char, r_char;
	      while((l_char=*l) && (r_char=*r))
		{
		  // check if this are digit characters
		  const bool l_digit=alphanum_isdigit(l_char), r_digit=alphanum_isdigit(r_char);
		  // if both characters are digits, we continue in NUMBER mode
		  if(l_digit && r_digit)
		    {
		      mode=NUMBER;
		      break;
		    }
		  // if only the left character is a digit, we have a result
		  if(l_digit) return -1;
		  // if only the right character is a digit, we have a result
		  if(r_digit) return +1;
		  // compute the difference of both characters
		  const int diff=l_char - r_char;
		  // if they differ we have a result
		  if(diff != 0) return diff;
		  // otherwise process the next characters
		  ++l;
		  ++r;
		}
	    }
	  else // mode==NUMBER
	    {
        long diff = 0;
        
#ifdef ALPHANUM_LOCALE
	      // get the left number
	      char *end;
	      unsigned long l_int=strtoul(l, &end, 0);
	      l=end;

	      // get the right number
	      unsigned long r_int=strtoul(r, &end, 0);
	      r=end;
#else
	 //      // get the left number
	 //      unsigned long l_int=0;
	 //      while(*l && alphanum_isdigit(*l))
		// {
		//   // TODO: this can overflow
		//   l_int=l_int*10 + *l-'0';
		//   ++l;
		// }

	 //      // get the right number
	 //      unsigned long r_int=0;
	 //      while(*r && alphanum_isdigit(*r))
		// {
		//   // TODO: this can overflow
		//   r_int=r_int*10 + *r-'0';
		//   ++r;
		// }
        int l_int_len = 0;
        const char* l_temp = l;
        while(*l && alphanum_isdigit(*l)) {
          l_int_len++;
          l++;
        }
        l = l_temp;

        int r_int_len = 0;
        const char* r_temp = r;
        while(*r && alphanum_isdigit(*r)) {
          r_int_len++;
          r++;
        }
        r = r_temp;

        diff = l_int_len - r_int_len;
        if (diff == 0) {
          for (int i = 0; i < l_int_len; i++, l++, r++) {
            int l_int = *l - '0';
            int r_int = *r - '0';
            diff = l_int - r_int;
            if (diff != 0) {
              return diff;
            }
          }
        }
#endif

	      // if the difference is not equal to zero, we have a comparison result
	      // const long diff=l_int-r_int;
	      if(diff != 0)
		return diff;

	      // otherwise we process the next substring in STRING mode
	      mode=STRING;
	    }
	}

      if(*r) return -1;
      if(*l) return +1;
      return 0;
    }

  }

  /**
     Compare left and right with the same semantics as strcmp(), but with the
     "Alphanum Algorithm" which produces more human-friendly
     results. The classes lT and rT must implement "std::ostream
     operator<< (std::ostream&, const Ty&)".

     @return negative if left<right, 0 if left==right, positive if left>right.
  */
  template <typename lT, typename rT>
  int alphanum_comp(const lT& left, const rT& right)
  {
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<" << typeid(left).name() << "," << typeid(right).name() << "> " << left << "," << right << std::endl;
#endif
    std::ostringstream l; l << left;
    std::ostringstream r; r << right;
    return alphanum_impl(l.str().c_str(), r.str().c_str());
  }

  /**
     Compare l and r with the same semantics as strcmp(), but with
     the "Alphanum Algorithm" which produces more human-friendly
     results.

     @return negative if l<r, 0 if l==r, positive if l>r.
  */
  template <>
  int alphanum_comp<std::string>(const std::string& l, const std::string& r)
  {
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<std::string,std::string> " << l << "," << r << std::endl;
#endif
    return alphanum_impl(l.c_str(), r.c_str());
  }

  ////////////////////////////////////////////////////////////////////////////

  // now follow a lot of overloaded alphanum_comp() functions to get a
  // direct call to alphanum_impl() upon the various combinations of c
  // and c++ strings.

  /**
     Compare l and r with the same semantics as strcmp(), but with
     the "Alphanum Algorithm" which produces more human-friendly
     results.

     @return negative if l<r, 0 if l==r, positive if l>r.
  */
  int alphanum_comp(char* l, char* r)
  {
    assert(l);
    assert(r);
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<char*,char*> " << l << "," << r << std::endl;
#endif
    return alphanum_impl(l, r);
  }

  int alphanum_comp(const char* l, const char* r)
  {
    assert(l);
    assert(r);
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<const char*,const char*> " << l << "," << r << std::endl;
#endif
    return alphanum_impl(l, r);
  }

  int alphanum_comp(char* l, const char* r)
  {
    assert(l);
    assert(r);
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<char*,const char*> " << l << "," << r << std::endl;
#endif
    return alphanum_impl(l, r);
  }

  int alphanum_comp(const char* l, char* r)
  {
    assert(l);
    assert(r);
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<const char*,char*> " << l << "," << r << std::endl;
#endif
    return alphanum_impl(l, r);
  }

  int alphanum_comp(const std::string& l, char* r)
  {
    assert(r);
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<std::string,char*> " << l << "," << r << std::endl;
#endif
    return alphanum_impl(l.c_str(), r);
  }

  int alphanum_comp(char* l, const std::string& r)
  {
    assert(l);
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<char*,std::string> " << l << "," << r << std::endl;
#endif
    return alphanum_impl(l, r.c_str());
  }

  int alphanum_comp(const std::string& l, const char* r)
  {
    assert(r);
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<std::string,const char*> " << l << "," << r << std::endl;
#endif
    return alphanum_impl(l.c_str(), r);
  }

  int alphanum_comp(const char* l, const std::string& r)
  {
    assert(l);
#ifdef DOJDEBUG
    std::clog << "alphanum_comp<const char*,std::string> " << l << "," << r << std::endl;
#endif
    return alphanum_impl(l, r.c_str());
  }

  ////////////////////////////////////////////////////////////////////////////

  /**
     Functor class to compare two objects with the "Alphanum
     Algorithm". If the objects are no std::string, they must
     implement "std::ostream operator<< (std::ostream&, const Ty&)".
  */
  template<class Ty>
  struct alphanum_less : public std::binary_function<Ty, Ty, bool>
  {
    bool operator()(const Ty& left, const Ty& right) const
    {
      return alphanum_comp(left, right) < 0;
    }
  };

}

#endif