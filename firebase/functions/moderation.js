// Pure helpers for the moderation links: actions are signed with an HMAC so
// only links from the moderator's email work.
const crypto = require("node:crypto");

const ACTIONS = ["keep", "remove", "ban"];

function sign(secret, groupId, postId, action) {
  return crypto.createHmac("sha256", secret).update(`${groupId}/${postId}/${action}`).digest("hex");
}

function verify(secret, groupId, postId, action, signature) {
  if (!ACTIONS.includes(action) || typeof signature !== "string") return false;
  const expected = Buffer.from(sign(secret, groupId, postId, action), "hex");
  const given = Buffer.from(signature, "hex");
  return given.length === expected.length && crypto.timingSafeEqual(given, expected);
}

function actionUrl(base, secret, groupId, postId, action) {
  const q = new URLSearchParams({ group: groupId, post: postId, action, sig: sign(secret, groupId, postId, action) });
  return `${base}?${q}`;
}

function reportEmail({ report, reportCount, links }) {
  const subject = `Scriberocks report: ${report.reason} (${reportCount} report${reportCount === 1 ? "" : "s"})`;
  const text = [
    `Group: ${report.groupId}`,
    `Post: ${report.postId} by ${report.authorId}`,
    `Reason: ${report.reason}`,
    `Reports on this post: ${reportCount}`,
    "",
    report.postText,
    "",
    `Keep (show it again): ${links.keep}`,
    `Remove: ${links.remove}`,
    `Remove and ban the author: ${links.ban}`,
  ].join("\n");
  return { subject, text };
}

module.exports = { ACTIONS, sign, verify, actionUrl, reportEmail };
