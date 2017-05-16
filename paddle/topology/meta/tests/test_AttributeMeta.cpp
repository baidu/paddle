#include <gtest/gtest.h>
#include <paddle/topology/meta/AttributeMeta.h>

TEST(AttributeMeta, MustSetLargerThan) {
  auto meta = paddle::topology::meta::AttributeMeta::create<int>(
      "size", "the size of this layer");
  meta->constraints<int>()->mustSet().largerThan(0);
  int test = 10;
  ASSERT_TRUE(meta->validate(&test, true).isOK());
  ASSERT_FALSE(meta->validate(&test, false).isOK());
  test = -1;
  ASSERT_FALSE(meta->validate(&test, true).isOK());
}

TEST(AttributeMeta, DefaultValueRnage) {
  auto meta = paddle::topology::meta::AttributeMeta::create<double>(
      "dropout", "the dropout rate of this layer");
  meta->constraints<double>()->defaultValue(0.0).inRange(0.0, 1.0);
  double test = -1.0;
  ASSERT_TRUE(meta->validate(&test, false).isOK());
  ASSERT_NEAR(test, 0.0, 1e-5);
  test = 1.2;
  ASSERT_FALSE(meta->validate(&test, true).isOK());
}
