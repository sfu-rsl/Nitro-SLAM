#include <graphite/vector.hpp>
#include <gtest/gtest.h>

namespace {

TEST(ManagedVectorTests, DefaultConstruction) {
  graphite::managed_vector<int> v;

  EXPECT_EQ(v.size(), 0u);
  EXPECT_EQ(v.capacity(), 0u);
  EXPECT_EQ(v.data().get(), nullptr);
}

TEST(ManagedVectorTests, PushBackAndPopBack) {
  graphite::managed_vector<int> v;

  v.push_back(10);
  v.push_back(20);
  v.push_back(30);

  ASSERT_EQ(v.size(), 3u);
  EXPECT_GE(v.capacity(), 3u);
  EXPECT_EQ(v[0], 10);
  EXPECT_EQ(v[1], 20);
  EXPECT_EQ(v[2], 30);
  EXPECT_EQ(v.back(), 30);

  v.pop_back();
  ASSERT_EQ(v.size(), 2u);
  EXPECT_EQ(v.back(), 20);

  v.pop_back();
  v.pop_back();
  v.pop_back(); // no-op on empty vector
  EXPECT_EQ(v.size(), 0u);
}

TEST(ManagedVectorTests, ReservePreservesElements) {
  graphite::managed_vector<int> v;
  v.push_back(3);
  v.push_back(7);

  v.reserve(16);

  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v.capacity(), 16u);
  EXPECT_EQ(v[0], 3);
  EXPECT_EQ(v[1], 7);
}

TEST(ManagedVectorTests, ResizeAndClear) {
  graphite::managed_vector<int> v;
  v.push_back(1);
  v.push_back(2);

  v.resize(8);
  EXPECT_EQ(v.size(), 8u);
  EXPECT_GE(v.capacity(), 8u);

  v.resize(1);
  ASSERT_EQ(v.size(), 1u);
  EXPECT_EQ(v[0], 1);

  const size_t cap_before_clear = v.capacity();
  v.clear();
  EXPECT_EQ(v.size(), 0u);
  EXPECT_EQ(v.capacity(), cap_before_clear);
}

TEST(ManagedVectorTests, DataBeginEndConsistency) {
  graphite::managed_vector<float> v;
  v.push_back(1.5f);
  v.push_back(2.5f);

  ASSERT_NE(v.data().get(), nullptr);
  EXPECT_EQ(v.begin(), v.data().get());
  EXPECT_EQ(v.end(), v.begin() + static_cast<ptrdiff_t>(v.size()));
  EXPECT_FLOAT_EQ(*(v.begin() + 1), 2.5f);
}

} // namespace
