const test = require("node:test");
const assert = require("node:assert");
const m = require("./moderation");

test("a signed action verifies; tampering doesn't", () => {
  const sig = m.sign("s3cret", "commons", "p1", "remove");
  assert.ok(m.verify("s3cret", "commons", "p1", "remove", sig));
  assert.ok(!m.verify("s3cret", "commons", "p1", "ban", sig), "different action");
  assert.ok(!m.verify("s3cret", "commons", "p2", "remove", sig), "different post");
  assert.ok(!m.verify("other", "commons", "p1", "remove", sig), "different secret");
  assert.ok(!m.verify("s3cret", "commons", "p1", "delete-everything", m.sign("s3cret", "commons", "p1", "delete-everything")), "unknown action");
  assert.ok(!m.verify("s3cret", "commons", "p1", "remove", "abc"), "malformed signature");
});

test("action links carry a verifiable signature", () => {
  const url = new URL(m.actionUrl("https://example.test/moderate", "k", "g", "p", "keep"));
  const q = url.searchParams;
  assert.strictEqual(q.get("action"), "keep");
  assert.ok(m.verify("k", q.get("group"), q.get("post"), q.get("action"), q.get("sig")));
});

test("the email names the reason, count, text and all three actions", () => {
  const { subject, text } = m.reportEmail({
    report: { groupId: "commons", postId: "p1", authorId: "u9", reason: "Hate", postText: "the post" },
    reportCount: 2,
    links: { keep: "K", remove: "R", ban: "B" },
  });
  assert.match(subject, /Hate \(2 reports\)/);
  for (const s of ["the post", "Keep (show it again): K", "Remove: R", "Remove and ban the author: B"]) assert.ok(text.includes(s), s);
});
