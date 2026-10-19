# Scriberocks moderation backend

Groups (forums) in the iOS app follow App Review Guideline 1.2: a content
filter before posting (on device), report and block on every post, published
contact (hello@sayists.com), and timely responses. Reports hide a post at once
for the reporter and for everyone at the group's threshold; this backend
emails each report to the moderator with signed **Keep**, **Remove** and
**Remove and ban** links.

The Firebase project `actuapedia-18ikj` is shared with other apps, so:

- **Rules:** merge `firestore-scriberocks.rules` into the project's existing
  rules in the console. Don't deploy it as the whole rules file.
- **Functions:** deploy from this folder. They're in their own codebase
  (`scriberocks`), so other apps' functions are left alone.

## One-time setup

1. Install the CLI and sign in: `npm install -g firebase-tools && firebase login`.
2. Secrets (iCloud mail: your Apple ID and an app-specific password from
   account.apple.com; mail is sent from hello@sayists.com):

       firebase functions:secrets:set SMTP_USER
       firebase functions:secrets:set SMTP_PASS
       firebase functions:secrets:set MODERATION_KEY   # any long random string, e.g. `openssl rand -hex 32`

3. Deploy: `cd functions && npm install && cd .. && firebase deploy --only functions:scriberocks`
4. Create groups in the console: collection `scriberocks_groups`, documents
   with `name` (string), `about` (string), `reportsToHide` (number: 1 for
   groups meant for younger writers, 2 otherwise).

## Tests

`cd functions && npm test` (signing and email content).
