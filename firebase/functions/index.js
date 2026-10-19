// Scriberocks moderation (Firebase project actuapedia-18ikj, shared):
//  - scriberocksReportEmail: emails each new report to the moderator with
//    signed Keep / Remove / Remove-and-ban links.
//  - scriberocksModerate: applies a signed action.
// Secrets: SMTP_USER, SMTP_PASS (iCloud: Apple ID + app-specific password),
// MODERATION_KEY (any long random string). See ../README.md.
const { onDocumentCreated } = require("firebase-functions/v2/firestore");
const { onRequest } = require("firebase-functions/v2/https");
const { defineSecret } = require("firebase-functions/params");
const { initializeApp } = require("firebase-admin/app");
const { getFirestore, FieldValue } = require("firebase-admin/firestore");
const nodemailer = require("nodemailer");
const m = require("./moderation");

initializeApp();
const db = getFirestore();

const SMTP_USER = defineSecret("SMTP_USER");
const SMTP_PASS = defineSecret("SMTP_PASS");
const MODERATION_KEY = defineSecret("MODERATION_KEY");
const MODERATOR = "hello@sayists.com";
const REGION = "us-central1";

exports.scriberocksReportEmail = onDocumentCreated(
  { document: "scriberocks_reports/{reportId}", region: REGION, secrets: [SMTP_USER, SMTP_PASS, MODERATION_KEY] },
  async (event) => {
    const report = event.data.data();
    const post = await db.doc(`scriberocks_groups/${report.groupId}/posts/${report.postId}`).get();
    const reportCount = post.exists ? (post.get("reportedBy") || []).length : 0;
    const base = `https://${REGION}-${process.env.GCLOUD_PROJECT}.cloudfunctions.net/scriberocksModerate`;
    const links = Object.fromEntries(m.ACTIONS.map((a) => [a, m.actionUrl(base, MODERATION_KEY.value(), report.groupId, report.postId, a)]));
    const { subject, text } = m.reportEmail({ report, reportCount, links });
    const transport = nodemailer.createTransport({
      host: "smtp.mail.me.com", port: 587, secure: false,
      auth: { user: SMTP_USER.value(), pass: SMTP_PASS.value() },
    });
    await transport.sendMail({ from: MODERATOR, to: MODERATOR, subject, text });
  },
);

exports.scriberocksModerate = onRequest({ region: REGION, secrets: [MODERATION_KEY] }, async (req, res) => {
  const { group, post, action, sig } = req.query;
  if (!m.verify(MODERATION_KEY.value(), group, post, action, sig)) {
    res.status(403).send("This link isn't valid.");
    return;
  }
  const ref = db.doc(`scriberocks_groups/${group}/posts/${post}`);
  const snap = await ref.get();
  if (!snap.exists) {
    res.send("Already removed.");
    return;
  }
  if (action === "keep") {
    await ref.update({ review: "kept" });
  } else {
    if (action === "ban") {
      await db.doc(`scriberocks_banned/${snap.get("authorId")}`).set({ bannedAt: FieldValue.serverTimestamp(), postId: post, groupId: group });
    }
    await ref.delete();
  }
  res.send({ keep: "Kept: the post is visible again.", remove: "Removed.", ban: "Removed, and the author can no longer post." }[action]);
});
