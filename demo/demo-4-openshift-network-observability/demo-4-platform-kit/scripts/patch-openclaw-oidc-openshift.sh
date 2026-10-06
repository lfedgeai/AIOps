#!/usr/bin/env bash
# Detach Kuadrant-generated AuthPolicies from OIDCPolicy and patch for OpenShift OAuth:
#   - scope=user:full (OpenShift rejects scope=openid)
#   - client_secret on authorization_code token exchange body only
#   - callback: no authentication block (Kuadrant pattern); access_token not id_token
#   - UI: drop JWT validation (OpenShift access tokens are opaque sha256~, not JWTs)
# Deletes OIDCPolicy afterward so the controller stops reverting patches.
set -euo pipefail

OPENCLAW_NS="${OPENCLAW_NS:-openclaw}"
KUBECTL="$(command -v oc || command -v kubectl)"
OAUTH_CLIENT_NAME="${OAUTH_CLIENT_NAME:-openclaw-rhcl}"
OAUTH_CLIENT_SECRET="${OAUTH_CLIENT_SECRET:-}"
OAUTH_JWKS_URL="${OAUTH_JWKS_URL:-}"
OIDC_POLICY_NAME="${OIDC_POLICY_NAME:-openclaw-ui-oidc}"
OPENCLAW_RHCL_HOST="${OPENCLAW_RHCL_HOST:-}"

[[ -n "$KUBECTL" ]] || { echo "oc/kubectl required" >&2; exit 1; }

if [[ -z "$OAUTH_CLIENT_SECRET" ]]; then
  OAUTH_CLIENT_SECRET="$("$KUBECTL" get oauthclient "$OAUTH_CLIENT_NAME" -o jsonpath='{.secret}' 2>/dev/null || true)"
fi
if [[ -z "$OAUTH_JWKS_URL" ]]; then
  api_server="$("$KUBECTL" whoami --show-server 2>/dev/null || true)"
  [[ -n "$api_server" ]] || { echo "could not resolve API server URL for JWKS" >&2; exit 1; }
  OAUTH_JWKS_URL="${api_server%/}/openid/v1/jwks"
fi
if [[ -z "$OPENCLAW_RHCL_HOST" ]]; then
  domain="$("$KUBECTL" get ingresscontroller default -n openshift-ingress-operator \
    -o jsonpath='{.status.domain}' 2>/dev/null || true)"
  [[ -n "$domain" ]] || { echo "could not resolve cluster ingress domain" >&2; exit 1; }
  OPENCLAW_RHCL_HOST="openclaw-rhcl.${domain}"
fi
[[ -n "$OAUTH_CLIENT_SECRET" ]] || { echo "OAuth client secret missing" >&2; exit 1; }

ingress_domain="${OPENCLAW_RHCL_HOST#openclaw-rhcl.}"
OAUTH_HOST="oauth-openshift.${ingress_domain}"
export OPENCLAW_NS OAUTH_CLIENT_SECRET OAUTH_JWKS_URL OPENCLAW_RHCL_HOST OAUTH_HOST KUBECTL
python3 <<'PY'
import json, os, subprocess, sys, time

ns = os.environ["OPENCLAW_NS"]
secret = os.environ["OAUTH_CLIENT_SECRET"]
rhcl_host = os.environ["OPENCLAW_RHCL_HOST"]
oauth_host = os.environ["OAUTH_HOST"]
kubectl = os.environ["KUBECTL"]

redirect_uri = f"https%3A%2F%2F{rhcl_host}%2Fauth%2Fcallback"
authorize_url = (
    f"https://{oauth_host}/oauth/authorize"
    f"?client_id=openclaw-rhcl&redirect_uri={redirect_uri}"
    f"&response_type=code&scope=user%3Afull"
)
base_url = f"https://{rhcl_host}"

# Query string from request.query or ?suffix of request.url_path (Envoy may omit request.query).
QUERY_CEL = (
    '(has(request.query) && request.query != "" ? request.query : '
    '(size(request.url_path.split("?")) > 1 ? request.url_path.split("?")[1] : ""))'
)
CODE_CEL = (
    f'{QUERY_CEL}.split("&").map(entry, entry.split("="))'
    '.filter(pair, pair[0] == "code").map(pair, pair[1])[0]'
)
CODE_PRESENT = (
    f'{QUERY_CEL}.split("&").map(entry, entry.split("="))'
    '.filter(pair, pair[0] == "code").map(pair, pair[1]).size() > 0'
)

TOKEN_BODY = (
    f'"code=" + {CODE_CEL} + "&grant_type=authorization_code&client_secret={secret}'
    f'&redirect_uri={redirect_uri}&client_id=openclaw-rhcl"'
)

COOKIE_PARSER = (
    "cookies := { name: value | raw_cookies := input.request.headers.cookie; "
    "cookie_parts := split(raw_cookies, \";\"); part := cookie_parts[_]; "
    "trimmed := trim(part, \" \"); eq_idx := indexof(trimmed, \"=\"); eq_idx != -1; "
    "name := trim(substring(trimmed, 0, eq_idx), \" \"); "
    "value := trim(substring(trimmed, eq_idx + 1, -1), \" \")}"
)

CALLBACK_OPA = f"""{COOKIE_PARSER}
location := concat("", ["{base_url}", cookies.target]) {{ input.auth.metadata.token.access_token; cookies.target }}
location := "{base_url}/" {{ input.auth.metadata.token.access_token; not cookies.target }}
location := "{authorize_url}" {{ not input.auth.metadata.token.access_token }}
allow = true"""

SESSION_OPA = f"""{COOKIE_PARSER}
allow = true {{ cookies.jwt != "" ; startswith(cookies.jwt, "sha256~") }}"""


def replace(doc: dict) -> None:
    subprocess.run(
        [kubectl, "replace", "-f", "-"],
        input=json.dumps(doc),
        text=True,
        check=True,
        capture_output=True,
    )


def patch_callback(doc: dict) -> None:
    rules = doc.setdefault("spec", {}).setdefault("overrides", {}).setdefault("rules", {})
    doc["spec"]["targetRef"] = {
        "group": "gateway.networking.k8s.io",
        "kind": "HTTPRoute",
        "name": "openclaw-ui-oidc-callback",
    }
    rules.pop("authentication", None)
    rules["authorization"] = {
        "location": {
            "metrics": False,
            "priority": 1,
            "opa": {"allValues": True, "rego": CALLBACK_OPA},
        },
        "deny": {
            "metrics": False,
            "priority": 2,
            "opa": {"allValues": False, "rego": "allow = false"},
        },
    }
    rules["metadata"] = {
        "token": {
            "metrics": False,
            "priority": 0,
            "when": [{"predicate": CODE_PRESENT}],
            "http": {
                "method": "POST",
                "url": f"https://{oauth_host}/oauth/token",
                "contentType": "application/x-www-form-urlencoded",
                "credentials": {},
                "body": {"expression": TOKEN_BODY},
            },
        }
    }
    rules["response"] = {
        "success": {},
        "unauthorized": {
            "code": 302,
            "headers": {
                "location": {"expression": "auth.authorization.location.location"},
                "set-cookie": {
                    "expression": (
                        f'"jwt=" + auth.metadata.token.access_token + '
                        f'"; domain={rhcl_host}; HttpOnly;  SameSite=Lax; Path=/; Max-Age=3600"'
                    )
                },
            },
        },
    }


def patch_ui(doc: dict) -> None:
    rules = doc.setdefault("spec", {}).setdefault("overrides", {}).setdefault("rules", {})
    # OpenShift session cookies are opaque sha256~ tokens — no JWT validation.
    rules.pop("authentication", None)
    rules["authorization"] = {
        "allow-openshift-session": {
            "metrics": False,
            "priority": 0,
            "opa": {"allValues": False, "rego": SESSION_OPA},
        }
    }
    resp = rules.setdefault("response", {})
    resp["success"] = {}
    for key in ("unauthenticated", "unauthorized"):
        resp[key] = {
            "code": 302,
            "headers": {
                "location": {"value": authorize_url},
                "set-cookie": {
                    "expression": (
                        f'"target=" + request.url_path + '
                        f'(has(request.query) && request.query != "" ? "?" + request.query : "") + '
                        f'"; domain={rhcl_host}; HttpOnly;  SameSite=Lax; Path=/; Max-Age=3600"'
                    )
                },
            },
        }


def ensure_callback_policy() -> None:
    exists = subprocess.run(
        [kubectl, "get", "authpolicy", "openclaw-ui-oidc-callback", "-n", ns],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode == 0:
        return
    doc = {
        "apiVersion": "kuadrant.io/v1",
        "kind": "AuthPolicy",
        "metadata": {
            "name": "openclaw-ui-oidc-callback",
            "namespace": ns,
            "labels": {"app.kubernetes.io/part-of": "netobserv-platform-merge"},
        },
        "spec": {
            "targetRef": {
                "group": "gateway.networking.k8s.io",
                "kind": "HTTPRoute",
                "name": "openclaw-ui-oidc-callback",
            },
            "overrides": {"strategy": "merge", "rules": {}},
        },
    }
    patch_callback(doc)
    subprocess.run(
        [kubectl, "apply", "-f", "-"],
        input=json.dumps(doc),
        text=True,
        check=True,
    )
    print("created authpolicy/openclaw-ui-oidc-callback")


def patch(name: str) -> None:
    for attempt in range(5):
        raw = subprocess.check_output([kubectl, "get", "authpolicy", name, "-n", ns, "-o", "json"], text=True)
        doc = json.loads(raw)
        doc.setdefault("metadata", {}).pop("ownerReferences", None)
        doc.pop("status", None)

        if name == "openclaw-ui-oidc-callback":
            patch_callback(doc)
        elif name == "openclaw-ui-oidc":
            patch_ui(doc)
        else:
            raise ValueError(f"unknown authpolicy {name}")

        try:
            replace(doc)
            return
        except subprocess.CalledProcessError as err:
            if attempt == 4:
                sys.stderr.write(err.stderr or "")
                raise
            time.sleep(2)


ensure_callback_policy()
for pol in ("openclaw-ui-oidc", "openclaw-ui-oidc-callback"):
    patch(pol)
    print(f"patched authpolicy/{pol}")

# openclaw-ui-authorize overrides openclaw-ui-oidc on the same HTTPRoute → 403 instead of OAuth redirect.
subprocess.run(
    [kubectl, "delete", "authpolicy", "openclaw-ui-authorize", "-n", ns, "--ignore-not-found"],
    check=False,
)
print("removed authpolicy/openclaw-ui-authorize (merged into openclaw-ui-oidc)")
PY

if "$KUBECTL" get oidcpolicy "$OIDC_POLICY_NAME" -n "$OPENCLAW_NS" >/dev/null 2>&1; then
  "$KUBECTL" delete oidcpolicy "$OIDC_POLICY_NAME" -n "$OPENCLAW_NS"
  echo "deleted oidcpolicy/$OIDC_POLICY_NAME (AuthPolicies now standalone)"
fi

# Re-apply ingress manifest without conflicting authorize policy if re-run from install script.
# Standalone patch: authorize already deleted above.
