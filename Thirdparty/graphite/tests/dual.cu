#include <cmath>
#include <graphite/dual.hpp>
#include <gtest/gtest.h>

namespace {
using Dualf = graphite::Dual<float, float>;

TEST(DualTests, ArithmeticOperators) {
  const Dualf a(3.0f, 2.0f);
  const Dualf b(4.0f, 5.0f);

  const auto sum = a + b;
  EXPECT_FLOAT_EQ(sum.real, 7.0f);
  EXPECT_FLOAT_EQ(sum.dual, 7.0f);

  const auto diff = a - b;
  EXPECT_FLOAT_EQ(diff.real, -1.0f);
  EXPECT_FLOAT_EQ(diff.dual, -3.0f);

  const auto prod = a * b;
  EXPECT_FLOAT_EQ(prod.real, 12.0f);
  EXPECT_FLOAT_EQ(prod.dual, 23.0f);

  const auto quot = a / b;
  EXPECT_FLOAT_EQ(quot.real, 0.75f);
  EXPECT_FLOAT_EQ(quot.dual, -0.4375f);
}

TEST(DualTests, CompoundAssignmentOperators) {
  Dualf x(3.0f, 2.0f);

  x += Dualf(4.0f, 5.0f);
  EXPECT_FLOAT_EQ(x.real, 7.0f);
  EXPECT_FLOAT_EQ(x.dual, 7.0f);

  x -= Dualf(1.0f, 2.0f);
  EXPECT_FLOAT_EQ(x.real, 6.0f);
  EXPECT_FLOAT_EQ(x.dual, 5.0f);

  x *= Dualf(2.0f, 3.0f);
  EXPECT_FLOAT_EQ(x.real, 12.0f);
  EXPECT_FLOAT_EQ(x.dual, 28.0f);

  x /= Dualf(4.0f, 1.0f);
  EXPECT_FLOAT_EQ(x.real, 3.0f);
  EXPECT_FLOAT_EQ(x.dual, 6.25f);
}

TEST(DualTests, ElementaryFunctions) {
  const Dualf x(0.5f, 2.0f);

  const auto s = sin(x);
  EXPECT_NEAR(s.real, std::sin(0.5f), 1e-6f);
  EXPECT_NEAR(s.dual, 2.0f * std::cos(0.5f), 1e-6f);

  const auto c = cos(x);
  EXPECT_NEAR(c.real, std::cos(0.5f), 1e-6f);
  EXPECT_NEAR(c.dual, -2.0f * std::sin(0.5f), 1e-6f);

  const auto e = exp(x);
  EXPECT_NEAR(e.real, std::exp(0.5f), 1e-6f);
  EXPECT_NEAR(e.dual, 2.0f * std::exp(0.5f), 1e-6f);

  const auto l = log(x);
  EXPECT_NEAR(l.real, std::log(0.5f), 1e-6f);
  EXPECT_NEAR(l.dual, 4.0f, 1e-6f);

  const auto r = sqrt(x);
  EXPECT_NEAR(r.real, std::sqrt(0.5f), 1e-6f);
  EXPECT_NEAR(r.dual, 2.0f / (2.0f * std::sqrt(0.5f)), 1e-6f);
}

TEST(DualTests, InverseTrigAndAbs) {
  const Dualf x(0.25f, 3.0f);

  const auto a = atan(x);
  EXPECT_NEAR(a.real, std::atan(0.25f), 1e-6f);
  EXPECT_NEAR(a.dual, 3.0f / (1.0f + 0.25f * 0.25f), 1e-6f);

  const auto ac = acos(x);
  EXPECT_NEAR(ac.real, std::acos(0.25f), 1e-6f);
  EXPECT_NEAR(ac.dual, -3.0f / std::sqrt(1.0f - 0.25f * 0.25f), 1e-6f);

  const Dualf neg(-2.0f, 4.0f);
  const auto ab = abs(neg);
  EXPECT_FLOAT_EQ(ab.real, 2.0f);
  EXPECT_FLOAT_EQ(ab.dual, -4.0f);
}

TEST(DualTests, ComparisonAndUnaryMinus) {
  const Dualf a(1.0f, 7.0f);
  const Dualf b(2.0f, -3.0f);

  EXPECT_TRUE(a < b);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(b >= a);

  const auto neg = -a;
  EXPECT_FLOAT_EQ(neg.real, -1.0f);
  EXPECT_FLOAT_EQ(neg.dual, -7.0f);
}

TEST(DualTests, DivisionByZeroReturnsInfinity) {
  const Dualf x(3.0f, 2.0f);
  const Dualf zero(0.0f, 1.0f);

  const auto y = x / zero;
  EXPECT_TRUE(std::isinf(y.real));
  EXPECT_TRUE(std::isinf(y.dual));

  Dualf z(1.0f, 1.0f);
  z /= zero;
  EXPECT_TRUE(std::isinf(z.real));
  EXPECT_TRUE(std::isinf(z.dual));
}

} // namespace
